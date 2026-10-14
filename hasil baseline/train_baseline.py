"""
Skrip Pelatihan Baseline YOLOv26 pada Dataset PlantDoc
Proyek Akhir Computer Vision - Teknik Komputer FILKOM UB

Deskripsi:
Melatih model dasar YOLOv26 (Nano / Small) untuk deteksi 30 kelas multi-penyakit
daun pada lingkungan alami (in-the-wild) sebelum integrasi Dynamic Attention.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def check_environment():
    print("=" * 60)
    print("Sistem & Environment Check")
    print("=" * 60)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Device     : {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("PERINGATAN: CUDA tidak terdeteksi, pelatihan akan berjalan di CPU!")
    print("=" * 60)


def train(args):
    check_environment()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"File konfigurasi dataset tidak ditemukan: {data_path}")

    print(f"\n[1] Menginisialisasi Model Baseline: {args.model}")
    model = YOLO(args.model)

    print(f"[2] Memulai Pelatihan:")
    print(f"    - Dataset    : {data_path}")
    print(f"    - Epochs     : {args.epochs}")
    print(f"    - Batch Size : {args.batch}")
    print(f"    - Image Size : {args.imgsz}")
    print(f"    - Device     : {args.device}")
    print(f"    - Output Dir : {args.project}/{args.name}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        amp=args.amp,
        patience=args.patience,
        save=True,
        plots=True,
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Pelatihan Selesai!")
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Bobot Terbaik (Best Checkpoint): {best_weight}")
    print(f"Grafik Hasil Evaluasi Disimpan di: {results.save_dir}")
    print("=" * 60)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv26 Baseline on PlantDoc")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Arsitektur baseline (yolo26n.pt / yolo26s.pt)")
    parser.add_argument("--data", type=str, default="plantdoc_data.yaml", help="Path file data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi citra")
    parser.add_argument("--device", type=str, default="0", help="Perangkat komputasi ('0' untuk GPU RTX 3060 atau 'cpu')")
    parser.add_argument("--workers", type=int, default=2, help="Jumlah worker dataloader (gunakan 2 di Windows)")
    parser.add_argument("--project", type=str, default="runs", help="Direktori penyimpanan eksperimen")
    parser.add_argument("--name", type=str, default="baseline_yolov26", help="Nama sub-folder eksperimen")
    parser.add_argument("--amp", action="store_true", default=True, help="Aktifkan Automatic Mixed Precision (FP16)")
    parser.add_argument("--patience", type=int, default=15, help="Patience untuk Early Stopping")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
