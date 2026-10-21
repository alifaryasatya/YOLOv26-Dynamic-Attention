"""Validate matched Phase 4 runs and report accuracy/cost against baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ultralytics import YOLO

import attention_runtime
from plantdoc_attention.integrations import P4AttentionWrapper
from common import dataset_content_sha256, load_yaml, resolve_path, sha256_file, write_json


MODELS = {
    "baseline": "baseline_pilot15_seed0",
    "bra_p4": "bra_p4_pilot15_seed0",
}


def suite_settings(dataset: str = "legacy", epochs: int = 50) -> dict:
    if dataset != "legacy":
        raise ValueError(f"Unknown dataset suite: {dataset}")
    if epochs not in (15, 50):
        raise ValueError("Expected a 15- or 50-epoch legacy suite")
    if epochs == 50:
        return {
            "data": "plantdoc_data.yaml",
            "models": {"baseline": "baseline_50e_seed0", "bra_p4": "bra_p4_50e_seed0"},
            "run_root": "runs/phase4_legacy_50e",
            "artifact_root": "artifacts/phase4_legacy_50e",
            "tag": "phase4_legacy_50e",
        }
    return {
        "data": "plantdoc_data.yaml",
        "models": MODELS.copy(),
        "run_root": "runs/phase4_legacy",
        "artifact_root": "artifacts/phase4_legacy",
        "tag": "phase4_legacy",
    }


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Artefak belum ada: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(manifests: dict[str, dict], epochs: int = 15) -> None:
    """Require complete matched runs; never compare the interim 15/50 baseline."""
    base = manifests["baseline"]
    excluded = {"project", "name"}
    recipe = {key: value for key, value in base["train_arguments"].items() if key not in excluded}
    for label, manifest in manifests.items():
        if manifest.get("status") != "completed" or manifest.get("completed_epochs") != epochs:
            raise ValueError(f"{label}: expected completed {epochs}-epoch training")
        if manifest.get("configured_epochs") != epochs or manifest["train_arguments"].get("epochs") != epochs:
            raise ValueError(f"{label}: scheduler must have been configured for {epochs} epochs")
        if manifest["train_arguments"].get("seed") != 0:
            raise ValueError(f"{label}: this pilot protocol requires seed 0")
        actual_recipe = {key: value for key, value in manifest["train_arguments"].items() if key not in excluded}
        if actual_recipe != recipe:
            changed = sorted(key for key in set(recipe) | set(actual_recipe) if recipe.get(key) != actual_recipe.get(key))
            raise ValueError(f"{label}: training recipe differs from baseline: {changed}")
        dataset_key = "dataset_content_sha256" if base.get("dataset_content_sha256") else "dataset_manifest_sha256"
        for key in ("model_source_sha256", dataset_key, "data_sha256", "implementation_sha256"):
            if not manifest.get(key) or manifest[key] != base.get(key):
                raise ValueError(f"{label}: missing or different provenance: {key}")
        for key in ("packages", "python"):
            if manifest["environment"].get(key) != base["environment"].get(key):
                raise ValueError(f"{label}: environment differs: {key}")
        attention = manifest.get("attention")
        if label == "baseline" and attention:
            raise ValueError("Baseline must not include an attention wrapper")
        if label != "baseline":
            kind = (attention or {}).get("kind", "bra")
            if not attention or kind != label.replace("_p4", "") or attention.get("layer") != 19:
                raise ValueError(f"{label}: incorrect attention kind or location")


def inspect_runs(run_root: Path, *, data: str = "plantdoc_data.yaml", models: dict | None = None,
                 epochs: int = 15) -> dict[str, dict]:
    models = MODELS if models is None else models
    manifests = {label: read_json(run_root / name / "experiment_manifest.json") for label, name in models.items()}
    validate_protocol(manifests, epochs=epochs)
    expected_source = sha256_file(resolve_path("yolo26n.pt"))
    expected_data = sha256_file(resolve_path(data))
    dataset_config, _ = load_yaml(data)
    expected_names = dataset_config["names"]
    if isinstance(expected_names, list):
        expected_names = dict(enumerate(expected_names))
    expected_content = dataset_content_sha256(data) if manifests["baseline"].get("dataset_content_sha256") else None
    for label, name in models.items():
        if manifests[label]["model_source_sha256"] != expected_source:
            raise ValueError(f"{label}: training must start from the agreed yolo26n.pt")
        if manifests[label]["data_sha256"] != expected_data:
            raise ValueError(f"{label}: dataset config differs from selected dataset: {data}")
        if expected_content is not None and manifests[label].get("dataset_content_sha256") != expected_content:
            raise ValueError(f"{label}: dataset files changed since training")
        run = run_root / name
        best = run / "weights/best.pt"
        if not best.is_file() or sha256_file(best) != manifests[label].get("best_checkpoint_sha256"):
            raise ValueError(f"{label}: best checkpoint missing or hash differs from manifest")
        model = YOLO(str(best)).model
        if model.model[-1].nc != dataset_config["nc"] or model.names != expected_names:
            raise ValueError(f"{label}: checkpoint class mapping differs from selected dataset")
        layer = model.model[19]
        if label == "baseline":
            if isinstance(layer, P4AttentionWrapper):
                raise ValueError("Baseline checkpoint unexpectedly contains attention")
        else:
            if not isinstance(layer, P4AttentionWrapper) or layer.config.get("kind", "bra") != label.replace("_p4", ""):
                raise ValueError(f"{label}: checkpoint attention differs from expected model")
            if not math.isfinite(float(layer.residual.gamma.detach())):
                raise ValueError(f"{label}: checkpoint gate is not finite")
        telemetry = read_json(run / "training_telemetry.json").get("epochs", [])
        if [item.get("epoch") for item in telemetry] != list(range(1, epochs + 1)):
            raise ValueError(f"{label}: telemetry must contain epochs 1..{epochs}")
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != epochs or [int(float(row["epoch"])) for row in rows] != list(range(1, epochs + 1)):
            raise ValueError(f"{label}: results.csv must contain epochs 1..{epochs}")
        if not all(math.isfinite(float(value)) for row in rows for value in row.values()):
            raise ValueError(f"{label}: results.csv contains NaN/Inf")
    return manifests


def compare_records(records: dict[str, dict]) -> list[dict]:
    baseline = records["baseline"]
    rows = []
    for label, record in records.items():
        evaluation, profile = record["evaluation"], record["profile"]
        for key in ("imgsz", "batch", "device", "dtype", "warmup_iterations", "measured_iterations"):
            if profile[key] != baseline["profile"][key]:
                raise ValueError(f"{label}: profiling protocol differs: {key}")
        base_gpu = baseline["profile"]["environment"].get("gpu")
        if profile["environment"].get("gpu") != base_gpu:
            raise ValueError(f"{label}: profiling GPU differs")
        for key in ("data_sha256", "split"):
            if evaluation[key] != baseline["evaluation"][key]:
                raise ValueError(f"{label}: evaluation differs: {key}")
        if evaluation["split"] != "val":
            raise ValueError("Phase 4 uses validation only")
        for key in ("imgsz", "batch", "device", "nms"):
            if evaluation["arguments"].get(key) != baseline["evaluation"]["arguments"].get(key):
                raise ValueError(f"{label}: evaluation settings differ: {key}")
        if evaluation["weights_sha256"] != profile["model_sha256"]:
            raise ValueError(f"{label}: evaluation and profile use different weights")
        metrics = evaluation["metrics"]
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"{label}: metrics contain NaN/Inf")
        base_metrics, base_profile = baseline["evaluation"]["metrics"], baseline["profile"]
        latency = profile["forward_latency_ms"]["mean_per_image"]
        base_latency = base_profile["forward_latency_ms"]["mean_per_image"]
        rows.append({
            "model": label, **metrics,
            "delta_map50": metrics["map50"] - base_metrics["map50"],
            "delta_map50_95": metrics["map50_95"] - base_metrics["map50_95"],
            "parameters": profile["parameters"], "gflops": profile["gflops"],
            "parameter_increase_pct": 100 * (profile["parameters"] / base_profile["parameters"] - 1),
            "forward_ms": latency, "latency_increase_pct": 100 * (latency / base_latency - 1),
            "inference_peak_allocated_mib": profile["peak_memory_mb"]["allocated"],
            "training_peak_allocated_mib": record.get("training_peak_allocated_mib"),
        })
    return rows


def make_report(run_root: Path, evaluation_root: Path, artifact_root: Path, tag: str,
                *, data: str = "plantdoc_data.yaml", models: dict | None = None, epochs: int = 15) -> Path:
    models = MODELS if models is None else models
    manifests = inspect_runs(run_root, data=data, models=models, epochs=epochs)
    records = {}
    for label, name in models.items():
        evaluation_dir = evaluation_root / f"{tag}_{label}_val"
        evaluation = read_json(evaluation_dir / "metrics_summary.json")
        profile = read_json(artifact_root / "profiles" / f"{tag}_{label}.json")
        best_hash = manifests[label]["best_checkpoint_sha256"]
        if evaluation["weights_sha256"] != best_hash:
            raise ValueError(f"{label}: evaluation does not use this run's best checkpoint")
        if evaluation["data_sha256"] != manifests[label]["data_sha256"]:
            raise ValueError(f"{label}: evaluation dataset differs from training")
        telemetry = read_json(run_root / name / "training_telemetry.json")["epochs"]
        peaks = [item["peak_allocated_mib"] for item in telemetry if item.get("peak_allocated_mib") is not None]
        records[label] = {"evaluation": evaluation, "profile": profile,
                          "training_peak_allocated_mib": max(peaks) if peaks else None,
                          "per_class_file": evaluation_dir / "per_class_metrics.csv"}
    rows = compare_records(records)
    output = artifact_root / f"comparison_{tag}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_class = {}
    for label, record in records.items():
        with record["per_class_file"].open(encoding="utf-8-sig", newline="") as handle:
            per_class[label] = {int(item["class_id"]): item for item in csv.DictReader(handle)}
    class_rows = []
    for class_id, base_class in per_class["baseline"].items():
        row = {"class_id": class_id, "class_name": base_class["class_name"]}
        for label in models:
            item = per_class[label][class_id]
            if item["class_name"] != base_class["class_name"]:
                raise ValueError("Per-class label mapping differs")
            row[f"{label}_map50_95"] = float(item["map50_95"])
        for label in (key for key in models if key != "baseline"):
            row[f"{label}_delta"] = row[f"{label}_map50_95"] - row["baseline_map50_95"]
        class_rows.append(row)
    with (output / "per_class_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    write_json(output / "comparison.json", {
        "status": "single_seed_pilot_complete" if epochs == 15 else "single_seed_comparison_complete",
        "epochs": epochs, "seed": 0,
        "data": data, "rows": rows, "training_manifests": manifests,
        "caveat": "One seed experiment; does not establish statistical significance or final test performance.",
    })
    lines = [f"# Fase 4 — Perbandingan {epochs} epoch, seed 0", "",
             "| Model | mAP50 | mAP50–95 | Delta mAP50–95 | Parameter | Forward ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
                     f"{row['delta_map50_95']:+.4f} | {row['parameters']:,} | {row['forward_ms']:.2f} |")
    lines += ["", "Hasil satu seed untuk screening. Tinjau AP per kelas dan biaya inferensi sebelum memilih desain.",
              "Klaim final memerlukan beberapa seed dan evaluasi test setelah desain dibekukan.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("legacy",), default="legacy")
    parser.add_argument("--epochs", type=int, choices=(15, 50), default=50)
    parser.add_argument("--run-root")
    parser.add_argument("--evaluation-root", default="runs/evaluation")
    parser.add_argument("--artifact-root")
    parser.add_argument("--tag")
    args = parser.parse_args()
    suite = suite_settings(args.dataset, args.epochs)
    try:
        output = make_report(resolve_path(args.run_root or suite["run_root"]), resolve_path(args.evaluation_root),
                             resolve_path(args.artifact_root or suite["artifact_root"]), args.tag or suite["tag"],
                             data=suite["data"], models=suite["models"], epochs=args.epochs)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(f"Comparison saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
