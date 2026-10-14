"""Run Phase 1 tests and a P4 CUDA backward check; write a JSON report."""

from __future__ import annotations

import argparse
import sys
import time
import unittest

import torch

from common import PROJECT_ROOT, environment_metadata, resolve_path, sha256_file, write_json

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from plantdoc_attention import ResidualAttention


def check_p4_cuda(batch: int) -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = ResidualAttention(128, num_heads=4, qk_dim=32, n_win=4, topk=4).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(batch, 128, 40, 40, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(x)
            loss = y.float().square().mean()
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        assert torch.isfinite(y).all()
        if step == 0:
            torch.testing.assert_close(y, x, atol=0, rtol=0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step == 1:
                assert parameter.grad.abs().sum() > 0, name
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {
        "status": "passed",
        "shape": [batch, 128, 40, 40],
        "amp_dtype": "float16",
        "optimizer_steps": 2,
        "finite_losses": losses,
        "gate_after_steps": float(model.gamma.detach()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "scope": "Standalone residual BRA forward/backward; excludes YOLO and dataloader.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/phase1/bra_validation.json")
    parser.add_argument("--batch", type=int, default=16, help="P4 CUDA backward batch")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_bra.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    p4 = {"status": "skipped", "reason": "CUDA unavailable"}
    if result.wasSuccessful() and torch.cuda.is_available():
        try:
            p4 = check_p4_cuda(args.batch)
        except Exception as exc:
            p4 = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    success = result.wasSuccessful() and p4["status"] != "failed"
    report = {
        "status": "passed" if success else "failed",
        "tests_run": result.testsRun,
        "failures": [(str(test), detail) for test, detail in result.failures],
        "errors": [(str(test), detail) for test, detail in result.errors],
        "skipped": [(str(test), reason) for test, reason in result.skipped],
        "p4_cuda_check": p4,
        "duration_seconds": time.perf_counter() - started,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in [
                PROJECT_ROOT / "src/plantdoc_attention/modules/bra.py",
                PROJECT_ROOT / "src/plantdoc_attention/modules/residual_attention.py",
                PROJECT_ROOT / "tests/test_bra.py",
                PROJECT_ROOT / "scripts/validate_bra.py",
            ]
        },
        "environment": environment_metadata(),
    }
    output = write_json(resolve_path(args.output), report)
    print(f"Phase 1 validation: {report['status']} | report: {output}")
    if p4["status"] == "passed":
        print(f"P4 batch {args.batch}: peak allocated {p4['peak_allocated_mib']:.2f} MiB")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
