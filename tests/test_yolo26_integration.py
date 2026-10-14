"""Integration tests on installed Ultralytics and local YOLO26 checkpoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plantdoc_attention.integrations import BRAConfig, BRADetectionTrainer, P4AttentionWrapper, attach_bra_p4
from plantdoc_attention.integrations.profiling import model_gflops


def assert_outputs_equal(test, first, second, *, atol=0, rtol=0):
    test.assertEqual(type(first), type(second))
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=atol, rtol=rtol)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_outputs_equal(test, first[key], second[key], atol=atol, rtol=rtol)
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_outputs_equal(test, a, b, atol=atol, rtol=rtol)
    else:
        test.assertEqual(first, second)


class YOLOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = YOLO(str(ROOT / "runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt")).model.cpu().float()

    def setUp(self):
        torch.manual_seed(0)

    def test_metadata_original_weights_and_gate_zero_identity(self):
        base = deepcopy(self.base).eval()
        custom = deepcopy(base)
        before = deepcopy(custom.state_dict())
        original = custom.model[19]
        save = deepcopy(custom.save)
        attach_bra_p4(custom)
        wrapper = custom.model[19]
        self.assertIs(wrapper.original, original)
        self.assertEqual(len(custom.model), 24)
        self.assertEqual(custom.save, save)
        self.assertEqual(wrapper.i, original.i)
        self.assertEqual(wrapper.f, original.f)
        self.assertEqual(custom.model[23].f, [16, 19, 22])
        self.assertEqual(wrapper.np, sum(p.numel() for p in wrapper.parameters()))
        for name, parameter in before.items():
            mapped = name.replace("model.19.", "model.19.original.") if name.startswith("model.19.") else name
            torch.testing.assert_close(parameter, custom.state_dict()[mapped], atol=0, rtol=0)
        with torch.no_grad():
            for size in ((128, 128), (128, 160)):
                x = torch.rand(1, 3, *size)
                assert_outputs_equal(self, base(x), custom(x))

    def test_refuses_duplicate_or_incompatible_wrapping(self):
        custom = deepcopy(self.base)
        attach_bra_p4(custom)
        with self.assertRaises(ValueError):
            attach_bra_p4(custom)
        for config in (BRAConfig(layer=16), BRAConfig(dim=64)):
            with self.assertRaises(ValueError):
                attach_bra_p4(deepcopy(self.base), config)

    def test_p4_output_reaches_detect_and_downstream_pan(self):
        base, custom = deepcopy(self.base).eval(), deepcopy(self.base).eval()
        attach_bra_p4(custom)
        with torch.no_grad():
            custom.model[19].residual.gamma.fill_(0.2)
        captured = [{}, {}]
        handles = []
        for index, network in enumerate((base, custom)):
            for layer in (16, 19, 22):
                def capture(module, inputs, output, index=index, layer=layer):
                    captured[index][layer] = output.detach().clone()
                handles.append(network.model[layer].register_forward_hook(capture))
        try:
            sample = torch.rand(1, 3, 128, 128)
            with torch.no_grad():
                base(sample)
                custom(sample)
            torch.testing.assert_close(captured[0][16], captured[1][16], atol=0, rtol=0)
            for layer in (19, 22):
                self.assertFalse(torch.equal(captured[0][layer], captured[1][layer]))
        finally:
            for handle in handles:
                handle.remove()

    def test_trainer_rebuild_preserves_base_and_attention_weights(self):
        # Use get_model in isolation; no dataloader, filesystem run, or training.
        trainer = BRADetectionTrainer.__new__(BRADetectionTrainer)
        trainer.attention_config = asdict(BRAConfig())
        trainer.args = SimpleNamespace(cls_remap=True)
        trainer.data = {"nc": 28, "channels": 3, "names": self.base.names}
        custom = trainer.get_model(cfg=self.base.yaml, weights=deepcopy(self.base), verbose=False)
        self.assertIsInstance(custom.model[19], P4AttentionWrapper)
        self.assertEqual(custom.model[-1].nc, 28)
        with torch.no_grad():
            custom.model[19].residual.gamma.fill_(0.2)
            custom.model[19].residual.attention.proj.weight.add_(0.05)
        rebuilt = trainer.get_model(cfg=custom.yaml, weights=custom, verbose=False)
        for name, value in custom.state_dict().items():
            torch.testing.assert_close(value, rebuilt.state_dict()[name], atol=0, rtol=0)
        # Check class-count remapping from COCO checkpoint also retains BRA.
        coco = YOLO(str(ROOT / "yolo26n.pt")).model
        remapped = trainer.get_model(cfg=coco.yaml, weights=coco, verbose=False)
        self.assertEqual(remapped.model[-1].nc, 28)
        self.assertIsInstance(remapped.model[19], P4AttentionWrapper)

    def test_config_matches_baseline_training_recipe(self):
        import yaml
        baseline = yaml.safe_load((ROOT / "configs/experiments/baseline.yaml").read_text())
        attention = yaml.safe_load((ROOT / "configs/experiments/bra_p4.yaml").read_text())
        for key in ("model", "data", "train", "evaluation", "profiling"):
            self.assertEqual(baseline[key], attention[key], key)

    def test_flops_include_routing_and_preserve_model(self):
        custom = deepcopy(self.base).eval()
        attach_bra_p4(custom)
        before_keys = set(custom.state_dict())
        base_flops = model_gflops(self.base, 128)
        custom_flops = model_gflops(custom, 128)
        self.assertGreater(custom_flops, base_flops)
        self.assertEqual(set(custom.state_dict()), before_keys)
        self.assertEqual(sum(p.numel() for p in custom.parameters()),
                         sum(p.numel() for p in self.base.parameters()) + 42817)

    def test_fusion_keeps_wrapper_and_identity(self):
        base, custom = deepcopy(self.base).eval(), deepcopy(self.base).eval()
        attach_bra_p4(custom)
        base.fuse(verbose=False)
        custom.fuse(verbose=False)
        self.assertIsInstance(custom.model[19], P4AttentionWrapper)
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            assert_outputs_equal(self, base(x), custom(x), atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_yolo_detection_loss_cuda_amp_backward(self):
        network = deepcopy(self.base).cuda().train().requires_grad_(True)
        attach_bra_p4(network)
        network.args = get_cfg(overrides={"epochs": 50})
        batch = {
            "img": torch.rand(2, 3, 128, 128, device="cuda"),
            "batch_idx": torch.tensor([0., 1.], device="cuda"),
            "cls": torch.tensor([[0.], [18.]], device="cuda"),
            "bboxes": torch.tensor([[.5, .5, .4, .4], [.4, .4, .2, .3]], device="cuda"),
        }
        for gate in (0.0, 0.1):
            with torch.no_grad():
                network.model[19].residual.gamma.fill_(gate)
            network.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = network.loss(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            residual = network.model[19].residual
            self.assertTrue(torch.isfinite(residual.gamma.grad))
            self.assertGreater(residual.gamma.grad.abs().item(), 0)
            for name, p in residual.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                if gate > 0:
                    self.assertGreater(p.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
