"""
Skrip Pengecekan Environment dan Integritas Dataset PlantDoc
Proyek Akhir Computer Vision - Teknik Komputer FILKOM UB
"""

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def check():
    print("=" * 60)
    print("VERIFIKASI LINGKUNGAN & DATASET PLANTDOC")
    print("=" * 60)
    
    # 1. Cek Python
    print(f"[1] Python Path     : {sys.executable}")
    print(f"    Python Version  : {sys.version.split()[0]}")

    # 2. Cek PyTorch & CUDA
    try:
        import torch
        print(f"[2] PyTorch Version : {torch.__version__}")
        print(f"    CUDA Available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU Device Name : {torch.cuda.get_device_name(0)}")
            print(f"    VRAM Kapasitas  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("    PERINGATAN: GPU CUDA tidak terdeteksi!")
    except ImportError:
        print("[2] PyTorch belum terpasang.")

    # 3. Cek Ultralytics
    try:
        import ultralytics
        print(f"[3] Ultralytics Ver : {ultralytics.__version__}")
    except ImportError:
        print("[3] Ultralytics belum terpasang.")

    # 4. Cek Konfigurasi Dataset
    yaml_file = Path("plantdoc_data.yaml")
    if yaml_file.exists():
        nc = 30
        base_path = Path("./archive")
        train_path = base_path / "train" / "images"
        val_path = base_path / "valid" / "images"
        test_path = base_path / "test" / "images"

        if yaml is not None:
            try:
                with open(yaml_file, "r") as f:
                    cfg = yaml.safe_load(f)
                base_path = Path(cfg.get("path", "./archive"))
                train_path = base_path / cfg.get("train", "train/images")
                val_path = base_path / cfg.get("val", "valid/images")
                test_path = base_path / cfg.get("test", "test/images")
                nc = cfg.get("nc", 30)
            except Exception:
                pass

        train_count = len(list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))) if train_path.exists() else 0
        val_count = len(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))) if val_path.exists() else 0
        test_count = len(list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))) if test_path.exists() else 0

        print(f"[4] Integritas Dataset PlantDoc:")
        print(f"    - Jumlah Kelas  : {nc} kelas")
        print(f"    - Train Images  : {train_count} citra ({'OK' if train_count > 0 else 'KOSONG'})")
        print(f"    - Valid Images  : {val_count} citra ({'OK' if val_count > 0 else 'KOSONG'})")
        print(f"    - Test Images   : {test_count} citra ({'OK' if test_count > 0 else 'KOSONG'})")
    else:
        print("[4] plantdoc_data.yaml belum ditemukan.")

    print("=" * 60)


if __name__ == "__main__":
    check()
