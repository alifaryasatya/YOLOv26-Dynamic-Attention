"""Entry point training reproducible untuk baseline dan eksperimen attention."""

from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

from ultralytics import YOLO
from attention_runtime import AttentionDetectionTrainer
from training_telemetry import TrainingTelemetry

from common import (
    PROJECT_ROOT,
    dataset_content_sha256,
    environment_metadata,
    load_experiment_config,
    load_yaml,
    resolve_model_reference,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 PlantDoc dari experiment config")
    parser.add_argument("--config", default="configs/experiments/baseline.yaml")
    parser.add_argument("--epochs", type=int, help="Override jumlah epoch")
    parser.add_argument("--device", help="Override device, contoh 0 atau cpu")
    parser.add_argument("--name", help="Override nama run")
    parser.add_argument("--batch", type=int, help="Override batch, for example 8 if GPU memory is insufficient")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved config without training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = load_experiment_config(args.config)
    experiment = dict(config.get("experiment", {}))
    train_args = dict(config.get("train", {}))

    model_reference = resolve_model_reference(config.get("model", "yolo26n.pt"))
    data_path = resolve_path(config.get("data", "plantdoc_data.yaml"))
    if not data_path.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {data_path}")

    project = resolve_path(experiment.get("project", "runs/experiments"))
    name = args.name or experiment.get("name", config_path.stem)
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    if args.device is not None:
        train_args["device"] = args.device
    if args.batch is not None:
        train_args["batch"] = args.batch
    train_args.update({"data": str(data_path), "project": str(project), "name": name})
    if int(train_args.get("epochs", 0)) < 1 or int(train_args.get("batch", 0)) < 1:
        raise ValueError("epochs and batch must be positive")
    if not train_args.get("exist_ok", False) and (project / name).exists():
        raise FileExistsError(f"Run already exists; choose a new --name: {project / name}")
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=train_args)  # Validate native Ultralytics arguments, without starting a trainer.
    dataset_config, _ = load_yaml(data_path)
    dataset_root = resolve_path(dataset_config.get("path", "."), data_path.parent)
    for split in ("train", "val"):
        split_path = resolve_path(dataset_config[split], dataset_root)
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset {split} split does not exist: {split_path}")

    print("=" * 72)
    print("TRAINING EXPERIMENT")
    print(f"Config : {config_path}")
    print(f"Model  : {model_reference}")
    print(f"Data   : {data_path}")
    print(f"Output : {project / name}")
    print(f"Epochs : {train_args.get('epochs')}")
    print(f"Seed   : {train_args.get('seed')}")
    print("=" * 72)

    if args.dry_run:
        import json
        print(json.dumps({"model": model_reference, "attention": config.get("attention"),
                          "train": train_args}, indent=2, ensure_ascii=False))
        print("DRY RUN: no model loaded, no training started.")
        return 0

    provenance = {
        "model_source_sha256": sha256_file(model_reference) if Path(model_reference).is_file() else None,
        "data_sha256": sha256_file(data_path),
        "dataset_manifest_sha256": sha256_file(dataset_root / "source_manifest.csv")
        if (dataset_root / "source_manifest.csv").is_file() else None,
        "implementation_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in [*sorted((PROJECT_ROOT / "src/plantdoc_attention").rglob("*.py")),
                         Path(__file__).resolve(), PROJECT_ROOT / "scripts/common.py",
                         PROJECT_ROOT / "scripts/training_telemetry.py"]
        },
    }
    if not provenance["dataset_manifest_sha256"]:
        print("Recording dataset image/label fingerprint before training (read-only)...", flush=True)
        provenance["dataset_content_sha256"] = dataset_content_sha256(data_path)
    model = YOLO(model_reference)
    attention_config = config.get("attention") or model.model.yaml.get("plantdoc_attention")
    trainer = partial(AttentionDetectionTrainer, attention_config=attention_config) if attention_config else None
    telemetry = TrainingTelemetry()
    model.add_callback("on_train_epoch_start", telemetry.start_epoch)
    model.add_callback("on_fit_epoch_end", telemetry.record_epoch)
    status = "completed"
    interrupted = False
    try:
        results = model.train(trainer=trainer, **train_args)
        save_dir = Path(results.save_dir).resolve()
    except KeyboardInterrupt:
        status = "interrupted"
        interrupted = True
        save_dir = Path(model.trainer.save_dir).resolve() if model.trainer else (project / name).resolve()
        if not save_dir.is_dir():
            raise

    completed_epochs = 0
    results_csv = save_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            completed_epochs = sum(1 for _ in csv.DictReader(handle))

    best_checkpoint = save_dir / "weights" / "best.pt"
    last_checkpoint = save_dir / "weights" / "last.pt"
    manifest = {
        "status": status,
        "completed_epochs": completed_epochs,
        "configured_epochs": train_args.get("epochs"),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data": str(data_path),
        **provenance,
        "model_reference": model_reference,
        "attention": attention_config,
        "resolved_experiment_config": config,
        "telemetry": str(save_dir / "training_telemetry.json"),
        "train_arguments": train_args,
        "description": experiment.get("description"),
        "command": [sys.executable, *sys.argv],
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": sha256_file(best_checkpoint) if best_checkpoint.is_file() else None,
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": sha256_file(last_checkpoint) if last_checkpoint.is_file() else None,
        "environment": environment_metadata(),
    }
    manifest_path = write_json(save_dir / "experiment_manifest.json", manifest)
    print(f"Manifest: {manifest_path}")
    if interrupted:
        print(f"Training dihentikan setelah {completed_epochs} epoch tersimpan.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
