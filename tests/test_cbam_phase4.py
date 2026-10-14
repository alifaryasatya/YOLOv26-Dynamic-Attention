"""CBAM module/integration and frozen Phase 4 recipe checks; no epoch training."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from plantdoc_attention.modules import CBAM, ResidualCBAM
from plantdoc_attention.integrations import CBAMConfig, AttentionDetectionTrainer, attach_attention_p4
from plantdoc_attention.integrations.profiling import model_gflops
from common import load_experiment_config
from test_yolo26_integration import assert_outputs_equal


class CBAMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_channel_then_spatial_matches_reference(self):
        module = CBAM(16, reduction_ratio=4).double()
        x = torch.randn(2, 16, 5, 7, dtype=torch.float64)
        first, last = module.channel_mlp[0], module.channel_mlp[2]
        def mlp(vector):
            hidden = F.linear(vector, first.weight.flatten(1), first.bias).relu()
            return F.linear(hidden, last.weight.flatten(1), last.bias)
        gate = (mlp(x.mean((2, 3))) + mlp(x.amax((2, 3)))).sigmoid()[:, :, None, None]
        weighted = x * gate
        features = torch.cat([weighted.mean(1, keepdim=True), weighted.amax(1, keepdim=True)], 1)
        expected = weighted * F.conv2d(features, module.spatial.weight, padding=3).sigmoid()
        torch.testing.assert_close(module(x), expected, rtol=1e-12, atol=1e-12)

    def test_shapes_gate_identity_and_gradients(self):
        for shape in ((2, 32, 7, 11), (1, 32, 1, 1)):
            module = ResidualCBAM(32, reduction_ratio=4)
            x = torch.randn(shape, requires_grad=True)
            torch.testing.assert_close(module(x), x, atol=0, rtol=0)
            module(x).square().mean().backward()
            self.assertGreater(module.gamma.grad.abs().item(), 0)
            for name, p in module.named_parameters():
                if name != "gamma":
                    self.assertEqual(p.grad.abs().sum().item(), 0)
            module.zero_grad(set_to_none=True)
            with torch.no_grad():
                module.gamma.fill_(.1)
            module(x).square().mean().backward()
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_invalid_parameters(self):
        for kwargs in ({"dim": 0}, {"dim": 16, "reduction_ratio": 0}, {"dim": 16, "spatial_kernel": 2}):
            with self.assertRaises(ValueError):
                CBAM(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_shape_dtype_and_backward(self):
        module = ResidualCBAM(128, gamma_init=.1).cuda()
        x = torch.randn(2, 128, 40, 40, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            y = module(x)
            loss = y.square().mean()
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        loss.backward()
        for name, p in module.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)


class Phase4IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = YOLO(str(ROOT / "runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt")).model.float().cpu()

    def test_matched_recipes(self):
        configs = [load_experiment_config(f"configs/experiments/phase4_{name}_15e.yaml")[0]
                   for name in ("baseline", "bra_p4", "cbam_p4")]
        for config in configs:
            self.assertEqual(config["train"]["epochs"], 15)
            self.assertEqual(config["train"]["close_mosaic"], 5)
            for key in ("model", "data", "train", "evaluation", "profiling"):
                self.assertEqual(config[key], configs[0][key], key)
        self.assertNotIn("attention", configs[0])
        self.assertEqual(configs[1]["attention"]["kind"], "bra")
        self.assertEqual(configs[2]["attention"]["kind"], "cbam")

    def test_identity_transfer_and_full_image_flops(self):
        base, custom = deepcopy(self.base).eval(), deepcopy(self.base).eval()
        original = custom.model[19]
        attach_attention_p4(custom, CBAMConfig())
        self.assertIs(custom.model[19].original, original)
        self.assertEqual(custom.model[-1].f, [16, 19, 22])
        x = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            assert_outputs_equal(self, base(x), custom(x))
        delta = sum(p.numel() for p in custom.parameters()) - sum(p.numel() for p in base.parameters())
        self.assertEqual(delta, 2539)
        self.assertGreater(model_gflops(custom, 128), model_gflops(base, 128))

    def test_trainer_rebuild_preserves_cbam_and_rejects_kind_change(self):
        trainer = AttentionDetectionTrainer.__new__(AttentionDetectionTrainer)
        trainer.attention_config = asdict(CBAMConfig())
        trainer.args = SimpleNamespace(cls_remap=True)
        trainer.data = {"nc": 28, "channels": 3, "names": self.base.names}
        custom = trainer.get_model(cfg=self.base.yaml, weights=deepcopy(self.base), verbose=False)
        with torch.no_grad():
            custom.model[19].residual.gamma.fill_(.125)
        rebuilt = trainer.get_model(cfg=custom.yaml, weights=custom, verbose=False)
        for name, value in custom.state_dict().items():
            torch.testing.assert_close(value, rebuilt.state_dict()[name], rtol=0, atol=0)
        trainer.attention_config = {"kind": "bra"}
        with self.assertRaises(ValueError):
            trainer.get_model(cfg=custom.yaml, weights=custom, verbose=False)

    def test_cbam_checkpoint_reload_and_fresh_process(self):
        yolo = YOLO(str(ROOT / "runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt"))
        attach_attention_p4(yolo.model, CBAMConfig(gamma_init=.125))
        reference = deepcopy(yolo.model).half().float().eval()
        parent = ROOT / "artifacts/tests/cbam_checkpoint"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory) / "cbam.pt"
            yolo.save(str(path))
            loaded = YOLO(str(path))
            self.assertEqual(loaded.model.model[19].config["kind"], "cbam")
            x = torch.rand(1, 3, 128, 128)
            with torch.no_grad():
                assert_outputs_equal(self, reference(x), loaded.model.eval()(x))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            code = (
                "import sys,torch; from ultralytics import YOLO; "
                "m=YOLO(sys.argv[1]); assert m.model.model[19].config['kind']=='cbam'; "
                "r=m.predict(torch.rand(1,3,128,128),imgsz=128,device='cpu',save=False,verbose=False); "
                "assert len(r)==1"
            )
            result = subprocess.run([sys.executable, "-c", code, str(path)], env=env,
                                    cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_yolo_cbam_detection_loss_amp_backward(self):
        network = deepcopy(self.base).cuda().train().requires_grad_(True)
        attach_attention_p4(network, CBAMConfig(gamma_init=.1))
        network.args = get_cfg(overrides={"epochs": 15})
        batch = {
            "img": torch.rand(2, 3, 128, 128, device="cuda"),
            "batch_idx": torch.tensor([0., 1.], device="cuda"),
            "cls": torch.tensor([[0.], [18.]], device="cuda"),
            "bboxes": torch.tensor([[.5, .5, .4, .4], [.4, .4, .2, .3]], device="cuda"),
        }
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = network.loss(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for name, parameter in network.model[19].residual.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
