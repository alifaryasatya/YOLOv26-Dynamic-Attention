"""
Skrip Inferensi & Visualisasi Prediksi Baseline YOLOv26
Proyek Akhir Computer Vision - Teknik Komputer FILKOM UB

Deskripsi:
Menjalankan deteksi pada sampel citra daun dengan background noise pekat (tanah, bayangan,
dedaunan tumpang tindih) untuk menghasilkan visualisasi bounding box baseline.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ultralytics import YOLO


def predict(args):
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"File bobot model tidak ditemukan: {weights_path}")

    source_path = Path(args.source).resolve()
    project_path = Path(args.project).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Sumber citra tidak ditemukan: {source_path}")

    print("=" * 60)
    print("Inferensi Visualisasi YOLOv26 / BRA-P4")
    print("=" * 60)
    print(f"Bobot Model : {weights_path}")
    print(f"Sumber Data : {source_path}")
    print(f"Confidence  : {args.conf}")
    print(f"Device      : {args.device}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(project_path),
        name=args.name,
        exist_ok=False,
        show_labels=True,
        show_conf=True
    )

    print("\n" + "=" * 60)
    print(f"Hasil visualisasi deteksi disimpan di: {results[0].save_dir}")
    print("=" * 60)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on Sample Images using YOLOv26 / BRA-P4")
    parser.add_argument("--weights", type=str, default="runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt", help="Path ke bobot best.pt")
    parser.add_argument("--source", type=str, default="dataset/valid/images", help="Path direktori atau file citra")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang batas confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="Ambang batas IoU")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra")
    parser.add_argument("--device", type=str, default="0", help="Perangkat komputasi ('0' atau 'cpu')")
    parser.add_argument("--project", type=str, default="runs/predict", help="Direktori output")
    parser.add_argument("--name", type=str, default="baseline_samples", help="Nama sub-folder output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)
