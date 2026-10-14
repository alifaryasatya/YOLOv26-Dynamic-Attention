"""Inspect completed Phase 3 results; never train or run inference."""

import argparse
import csv
import json
import math
import statistics

from ultralytics import YOLO

import attention_runtime  # Make custom checkpoint classes importable.
from plantdoc_attention.integrations import P4AttentionWrapper
from common import resolve_path, write_json


def assess(rows: list[dict], telemetry: list[dict], stage: str, expected_epochs: int) -> dict:
    issues, warnings = [], []
    if len(rows) != expected_epochs:
        issues.append(f"Expected {expected_epochs} saved epochs, found {len(rows)}")
    loss_keys = [key for key in (rows[0] if rows else {}) if key.startswith("train/") and key.endswith("loss")]
    if not loss_keys:
        issues.append("Training loss columns missing")
    losses = []
    for row in rows:
        try:
            values = [float(row[key]) for key in loss_keys]
            all_numbers = [float(value) for key, value in row.items() if key != "epoch"]
            if not all(math.isfinite(value) for value in all_numbers):
                issues.append(f"Non-finite CSV value at epoch {row.get('epoch')}")
            losses.append(sum(values))
        except (TypeError, ValueError):
            issues.append(f"Non-numeric CSV value at epoch {row.get('epoch')}")
    ratio = None
    if losses and all(math.isfinite(value) for value in losses):
        window = min(3, max(1, len(losses) // 2))
        initial = statistics.fmean(losses[:window])
        final = statistics.fmean(losses[-window:])
        ratio = final / initial if initial > 0 else None
        if stage == "overfit" and (ratio is None or ratio > 0.8):
            warnings.append("Overfit loss did not drop by at least 20%; inspect samples or extend the diagnostic run")
        if stage == "screening" and ratio is not None and ratio > 1.1:
            warnings.append("Late training loss exceeds initial loss by more than 10%")
    if len(telemetry) != len(rows):
        issues.append("Missing telemetry for one or more saved epochs")
    if [sample.get("epoch") for sample in telemetry] != [int(float(row["epoch"])) for row in rows]:
        issues.append("Telemetry epochs do not match results.csv")
    for sample in telemetry:
        gamma = sample.get("gamma")
        if gamma is None or not math.isfinite(gamma):
            issues.append(f"Missing/non-finite BRA gate at epoch {sample.get('epoch')}")
        allocated = sample.get("peak_allocated_mib")
        total = sample.get("gpu_total_mib")
        if allocated is not None and total and allocated / total > .9:
            warnings.append(f"Epoch {sample['epoch']}: allocated VRAM exceeds 90% of GPU memory")
    if telemetry and telemetry[-1].get("gamma") == 0:
        warnings.append("BRA gate is still zero at the last epoch; inspect optimizer/gradients")
    peaks = [item["peak_allocated_mib"] for item in telemetry if item.get("peak_allocated_mib") is not None]
    if len(peaks) >= 3 and peaks[-1] - peaks[1] > 256 and peaks[-1] > 1.1 * peaks[1]:
        warnings.append("Allocated VRAM grew after warm-up; review memory behavior before longer training")
    return {
        "status": "failed" if issues else "needs_review" if warnings else "passed",
        "stage": stage, "saved_epochs": len(rows), "expected_epochs": expected_epochs,
        "summed_loss_final_to_initial_ratio": ratio,
        "peak_allocated_mib": max(peaks) if peaks else None,
        "final_training_gate": telemetry[-1].get("gamma") if telemetry else None,
        "issues": issues, "warnings": warnings,
        "note": "Diagnostic heuristics; GPU peaks alone cannot establish absence of memory leaks.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", required=True, choices=("overfit", "smoke", "screening"))
    parser.add_argument("--output", help="Default: RUN/phase3_review.json")
    args = parser.parse_args()
    run = resolve_path(args.run)
    required = [run / "results.csv", run / "experiment_manifest.json", run / "training_telemetry.json",
                run / "weights/best.pt", run / "weights/last.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Run belum lengkap; file yang belum ada:\n" + "\n".join(missing))
    with required[0].open(encoding="utf-8-sig", newline="") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    manifest = json.loads(required[1].read_text())
    telemetry = json.loads(required[2].read_text())["epochs"]
    report = assess(rows, telemetry, args.stage, int(manifest["configured_epochs"]))
    model = YOLO(str(required[3]))  # Check serialized class/metadata only; no forward pass.
    layer = model.model.model[19]
    if not isinstance(layer, P4AttentionWrapper) or model.model.model[-1].nc != 28:
        report["issues"].append("Checkpoint does not contain 28-class YOLO BRA-P4")
        report["status"] = "failed"
    elif not math.isfinite(float(layer.residual.gamma.detach())):
        report["issues"].append("Checkpoint gate is not finite")
        report["status"] = "failed"
    report["manifest_status"] = manifest["status"]
    if manifest["status"] != "completed":
        report["issues"].append("Training manifest is not completed")
        report["status"] = "failed"
    report["validation_last_epoch"] = {key: float(value) for key, value in rows[-1].items()
                                       if key.startswith("metrics/")} if rows else {}
    path = write_json(resolve_path(args.output) if args.output else run / "phase3_review.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
