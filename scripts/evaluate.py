"""Evaluasi terstruktur YOLO26 pada validation atau test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params
from attention_runtime import model_gflops

from common import environment_metadata, resolve_model_reference, resolve_path, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint YOLO26 PlantDoc")
    parser.add_argument("--weights", required=True, help="Checkpoint .pt")
    parser.add_argument("--data", default="plantdoc_data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true", help="Konfirmasi bahwa desain telah dibekukan")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--name", default="baseline_val")
    parser.add_argument("--nms", choices=("true", "false"), help="Override head inference YOLO26")
    return parser.parse_args()


def value_at(values: Any, index: int) -> float | None:
    if values is None:
        return None
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test evaluation. Tambahkan --allow-test setelah desain model dibekukan.")

    weights = resolve_path(args.weights)
    data = resolve_path(args.data)
    project = resolve_path(args.project)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {data}")

    model = YOLO(resolve_model_reference(weights))
    val_args: dict[str, Any] = {
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(project),
        "name": args.name,
        "plots": True,
        "verbose": True,
        "exist_ok": False,
    }
    if args.nms is not None:
        val_args["nms"] = args.nms == "true"

    metrics = model.val(**val_args)
    save_dir = Path(metrics.save_dir).resolve()
    box = metrics.box
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    per_class_path = save_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "precision", "recall", "ap50", "map50_95"])
        for class_id in range(len(names)):
            writer.writerow(
                [
                    class_id,
                    names[class_id],
                    value_at(box.p, class_id),
                    value_at(box.r, class_id),
                    value_at(box.ap50, class_id),
                    value_at(box.maps, class_id),
                ]
            )

    speed = {key: float(value) for key, value in metrics.speed.items()}
    latency = speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
    summary = {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "data": str(data),
        "data_sha256": sha256_file(data),
        "split": args.split,
        "arguments": val_args,
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "speed_ms_per_image": speed,
        "total_latency_ms_per_image": latency,
        "fps": 1000.0 / latency if latency > 0 else None,
        "parameters": get_num_params(model.model),
        "gflops": model_gflops(model.model, args.imgsz),
        "save_dir": str(save_dir),
        "per_class_metrics": str(per_class_path),
        "environment": environment_metadata(),
    }
    summary_path = write_json(save_dir / "metrics_summary.json", summary)
    print("=" * 72)
    print(f"mAP50       : {box.map50:.4f}")
    print(f"mAP50-95    : {box.map:.4f}")
    print(f"Precision   : {box.mp:.4f}")
    print(f"Recall      : {box.mr:.4f}")
    print(f"Summary     : {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
