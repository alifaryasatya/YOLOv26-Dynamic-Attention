"""Phase 2 integration tests and one real PlantDoc forward/backward batch."""

from __future__ import annotations

import argparse
import sys
import time
import unittest

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

from attention_runtime import attach_bra_p4
from common import PROJECT_ROOT, environment_metadata, load_yaml, resolve_path, sha256_file, write_json


def real_batch_check(weights: str, data_path: str, device: str) -> dict:
    config, config_path = load_yaml(data_path)
    dataset_root = (config_path.parent / config["path"]).resolve()
    dataset = YOLODataset(img_path=str(dataset_root / config["train"]), data=config,
                          imgsz=640, batch_size=2, augment=False, rect=False, hyp=get_cfg())
    # This exercises actual Ultralytics image/label transforms and collation.
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    target_device = torch.device(device)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(target_device)
    batch["img"] = batch["img"].float() / 255
    network = YOLO(str(resolve_path(weights))).model.to(target_device).float().train().requires_grad_(True)
    attach_bra_p4(network)
    network.args = get_cfg(overrides={"epochs": 50})
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast(target_device.type, dtype=torch.float16, enabled=target_device.type == "cuda"):
        loss, components = network.loss(batch)
    if not torch.isfinite(loss).all():
        raise AssertionError("Detection loss is not finite")
    loss.sum().backward()
    gate = network.model[19].residual.gamma
    if gate.grad is None or not torch.isfinite(gate.grad) or gate.grad.abs() == 0:
        raise AssertionError("No finite nonzero gradient on residual gate")
    return {
        "status": "passed", "shape": list(batch["img"].shape),
        "images": list(batch["im_file"]), "objects": len(batch["cls"]),
        "total_loss": float(loss.detach().sum()),
        "loss_components": {key: float(value) for key, value in components.items()},
        "gate_gradient": float(gate.grad),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if target_device.type == "cuda" else None,
        "amp": target_device.type == "cuda",
        "scope": "One real train-split batch; no optimizer step, no epoch training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt")
    parser.add_argument("--data", default="plantdoc_data.yaml")
    parser.add_argument("--output", default="artifacts/integration_legacy/integration_validation.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    batch_report = {"status": "not_run"}
    if result.wasSuccessful():
        try:
            batch_report = real_batch_check(args.weights, args.data, args.device)
        except Exception as exc:
            batch_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    success = result.wasSuccessful() and batch_report["status"] == "passed"
    source_files = [*sorted((PROJECT_ROOT / "src/plantdoc_attention").rglob("*.py")),
                    *sorted((PROJECT_ROOT / "tests").glob("test_*.py")),
                    PROJECT_ROOT / "scripts/validate_yolo_attention.py",
                    PROJECT_ROOT / "scripts/train.py", PROJECT_ROOT / "scripts/evaluate.py",
                    PROJECT_ROOT / "scripts/benchmark_model.py",
                    PROJECT_ROOT / "scripts/attention_runtime.py",
                    PROJECT_ROOT / "scripts/create_attention_model.py",
                    PROJECT_ROOT / "predict_sample.py",
                    PROJECT_ROOT / "configs/experiments/bra_p4.yaml"]
    report = {
        "status": "passed" if success else "failed", "tests_run": result.testsRun,
        "failures": [(str(test), detail) for test, detail in result.failures],
        "errors": [(str(test), detail) for test, detail in result.errors],
        "skipped": [(str(test), reason) for test, reason in result.skipped],
        "real_batch": batch_report,
        "duration_seconds": time.perf_counter() - started,
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_files},
        "environment": environment_metadata(),
    }
    output = write_json(resolve_path(args.output), report)
    print(f"Phase 2 integration: {report['status']} | report: {output}")
    if not success:
        print(batch_report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
