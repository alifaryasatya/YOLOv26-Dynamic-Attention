"""Run: .venv/Scripts/python.exe -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantdoc_attention import BiLevelRoutingAttention, ResidualAttention
from plantdoc_attention.modules.bra import _partition, _unpartition


class BRATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def assert_finite_gradients(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_partition_roundtrip_and_spatial_order(self) -> None:
        x = torch.arange(24.0).reshape(1, 1, 4, 6)
        tokens = _partition(x, 2, 3)
        torch.testing.assert_close(tokens[0, 0, :, 0], torch.tensor([0., 1., 2., 6., 7., 8.]))
        torch.testing.assert_close(_unpartition(tokens, 2, 2, 2, 3), x)

    def test_shapes_nondivisible_small_rectangular_and_noncontiguous(self) -> None:
        model = BiLevelRoutingAttention(16, qk_dim=8)
        for shape in ((2, 16, 8, 12), (1, 16, 7, 11), (1, 16, 1, 1), (1, 16, 2, 3)):
            with self.subTest(shape=shape):
                x = torch.randn(shape).transpose(2, 3)
                y, routing = model(x, return_routing=True)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(y.device, x.device)
                self.assertTrue(torch.isfinite(y).all())
                self.assertFalse(routing.requires_grad)
                self.assertLess(routing.max().item(), routing.shape[1])

    def test_p4_shape(self) -> None:
        model = BiLevelRoutingAttention(128, qk_dim=32)
        with torch.no_grad():
            y = model(torch.randn(1, 128, 40, 40))
        self.assertEqual(y.shape, (1, 128, 40, 40))

    def test_backward_reaches_qkv_projection_and_local_context(self) -> None:
        model = BiLevelRoutingAttention(16, qk_dim=8)
        x = torch.randn(2, 16, 7, 11, requires_grad=True)
        model(x).square().mean().backward()
        self.assert_finite_gradients(model)
        self.assertTrue(torch.isfinite(x.grad).all())
        for section in model.qkv.weight.grad.split((8, 8, 16), dim=0):
            self.assertGreater(section.abs().sum().item(), 0)

    def test_all_regions_equals_dense_attention_including_padding(self) -> None:
        # Independent dense oracle: padded keys must have exactly zero influence.
        model = BiLevelRoutingAttention(8, num_heads=2, qk_dim=4, n_win=2,
                                       topk=4, side_dwconv=0).double().eval()
        x = torch.randn(2, 8, 3, 5, dtype=torch.float64)
        q, k, v = model.qkv(x).split((4, 4, 8), dim=1)
        batch, _, height, width = x.shape
        q = q.reshape(batch, 2, 2, -1).transpose(-1, -2)
        k = k.reshape(batch, 2, 2, -1)
        v = v.reshape(batch, 2, 4, -1).transpose(-1, -2)
        dense = ((q * (2 ** -0.5)) @ k).softmax(-1) @ v
        dense = dense.transpose(-1, -2).reshape(batch, 8, height, width)
        expected = model.proj(dense)
        torch.testing.assert_close(model(x), expected, atol=1e-10, rtol=1e-10)

    def test_routing_is_content_dependent(self) -> None:
        model = BiLevelRoutingAttention(4, num_heads=1, n_win=2, topk=1, side_dwconv=0)
        with torch.no_grad():
            model.qkv.weight.zero_()
            model.qkv.bias.zero_()
            model.qkv.weight[0, 0, 0, 0] = 1
            model.qkv.weight[4, 0, 0, 0] = 1
        x = torch.ones(1, 4, 4, 4)
        x[:, 0, :2, :2] = 10
        _, first = model(x, return_routing=True)
        x[:, 0, :2, :2] = 1
        x[:, 0, 2:, 2:] = 10
        _, second = model(x, return_routing=True)
        self.assertTrue((first == 0).all())
        self.assertTrue((second == 3).all())

    def test_only_selected_regions_supply_values(self) -> None:
        model = BiLevelRoutingAttention(4, num_heads=1, n_win=2, topk=1, side_dwconv=0)
        with torch.no_grad():
            model.qkv.weight.zero_(); model.qkv.bias.zero_()
            model.qkv.bias[:4] = 1  # constant queries
            model.qkv.weight[4:8, 0, 0, 0] = 1  # keys depend on channel 0
            model.qkv.weight[8:12, 1, 0, 0] = 1  # values depend on channel 1
            model.proj.weight.copy_(torch.eye(4).reshape(4, 4, 1, 1))
            model.proj.bias.zero_()
        x = torch.ones(1, 4, 4, 4)
        x[:, 0, :2, :2] = 10
        x[:, 1, :2, :2] = 3
        before = model(x)
        x[:, 1, 2:, :] = 1000
        torch.testing.assert_close(model(x), before)
        torch.testing.assert_close(before, torch.full_like(before, 3))

    def test_zero_gate_identity_and_branch_activation(self) -> None:
        model = ResidualAttention(8, num_heads=2, qk_dim=4, n_win=2, topk=2)
        x = torch.randn(1, 8, 5, 7, requires_grad=True)
        torch.testing.assert_close(model(x), x, atol=0, rtol=0)
        model(x).square().mean().backward()
        self.assertNotEqual(model.gamma.grad.item(), 0)
        torch.testing.assert_close(x.grad, 2 * x.detach() / x.numel())
        for name, parameter in model.named_parameters():
            if name != "gamma":
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(parameter.grad.abs().sum().item(), 0, name)
        with torch.no_grad():
            model.gamma -= 0.1 * model.gamma.grad
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        self.assert_finite_gradients(model)

    def test_gradcheck_without_discrete_routing_changes(self) -> None:
        model = BiLevelRoutingAttention(4, num_heads=2, qk_dim=2, n_win=1,
                                       topk=1, side_dwconv=0).double()
        x = torch.randn(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model, (x,), fast_mode=True))

    def test_eval_determinism_and_state_dict_reload(self) -> None:
        model = ResidualAttention(8, gamma_init=0.1, num_heads=2,
                                  n_win=2, topk=2, attn_drop=0.3).eval()
        other = ResidualAttention(8, gamma_init=0.1, num_heads=2,
                                  n_win=2, topk=2, attn_drop=0.3).eval()
        other.load_state_dict(model.state_dict())
        x = torch.randn(1, 8, 5, 7)
        torch.testing.assert_close(model(x), model(x), atol=0, rtol=0)
        torch.testing.assert_close(model(x), other(x), atol=0, rtol=0)

    def test_invalid_configuration_and_input(self) -> None:
        for args in (dict(dim=0), dict(dim=8, qk_dim=3), dict(dim=7),
                     dict(dim=8, topk=17), dict(dim=8, n_win=0),
                     dict(dim=8, side_dwconv=2), dict(dim=8, attn_drop=1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                BiLevelRoutingAttention(**args)
        model = BiLevelRoutingAttention(8)
        for x in (torch.randn(8, 4, 4), torch.randn(1, 4, 4, 4), torch.randn(1, 8, 0, 4)):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(TypeError):
            model(torch.ones(1, 8, 4, 4, dtype=torch.int64))

    def test_cpu_bfloat16_amp(self) -> None:
        model = ResidualAttention(16, qk_dim=8, gamma_init=0.1)
        x = torch.randn(1, 16, 5, 7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = model(x)
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assert_finite_gradients(model)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp32_fp16_and_amp(self) -> None:
        for dtype, amp in ((torch.float32, False), (torch.float16, False), (torch.float32, True)):
            with self.subTest(dtype=dtype, amp=amp):
                model = ResidualAttention(32, qk_dim=8, gamma_init=0.1).cuda().to(dtype)
                x = torch.randn(2, 32, 7, 11, device="cuda", dtype=dtype, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    y = model(x)
                    loss = y.float().square().mean()
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(y.device, x.device)
                self.assertTrue(torch.isfinite(y).all())
                loss.backward()
                self.assert_finite_gradients(model)
                self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
