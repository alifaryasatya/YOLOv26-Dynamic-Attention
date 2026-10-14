"""
Skrip Evaluasi Baseline YOLOv26 pada Dataset Pengujian PlantDoc
Proyek Akhir Computer Vision - Teknik Komputer FILKOM UB

Deskripsi:
Menghitung metrik performa deteksi (mAP@0.5, mAP@0.5:0.95, Precision, Recall, Latensi inferensi)
pada split data uji (test set) untuk baseline YOLOv26.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate(args):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"File bobot model tidak ditemukan: {weights_path}")

    print("=" * 60)
    print("Evaluasi Model Baseline YOLOv26 pada Test Set PlantDoc")
    print("=" * 60)
    print(f"Bobot Model : {weights_path}")
    print(f"Dataset      : {args.data}")
    print(f"Split        : {args.split}")
    print(f"Device       : {args.device}")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("HASIL METRIK EVALUASI")
    print("=" * 60)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    
    speed = metrics.speed
    preprocess = speed.get('preprocess', 0)
    inference = speed.get('inference', 0)
    loss = speed.get('loss', 0)
    postprocess = speed.get('postprocess', 0)
    total_ms = preprocess + inference + postprocess
    fps = 1000.0 / total_ms if total_ms > 0 else 0

    print("-" * 60)
    print("LATENSI & KECEPATAN (EDGE COMPUTING METRICS):")
    print(f"Pre-process  : {preprocess:.2f} ms")
    print(f"Inference    : {inference:.2f} ms (NMS-Free Native YOLOv26)")
    print(f"Post-process : {postprocess:.2f} ms")
    print(f"Total Latency: {total_ms:.2f} ms per image (~{fps:.1f} FPS)")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv26 Baseline on PlantDoc Test Set")
    parser.add_argument("--weights", type=str, default="runs/detect/yolov26_baseline/weights/best.pt", help="Path ke bobot best.pt")
    parser.add_argument("--data", type=str, default="plantdoc_data.yaml", help="Path konfigurasi dataset")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Data split untuk evaluasi")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi citra")
    parser.add_argument("--batch", type=int, default=16, help="Batch size evaluasi")
    parser.add_argument("--device", type=str, default="0", help="Perangkat komputasi ('0' atau 'cpu')")
    parser.add_argument("--project", type=str, default="runs/val", help="Direktori output evaluasi")
    parser.add_argument("--name", type=str, default="yolov26_baseline_test", help="Nama sub-folder evaluasi")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
