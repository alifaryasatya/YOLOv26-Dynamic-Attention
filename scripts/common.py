"""Utilitas bersama untuk pipeline eksperimen PlantDoc."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Membaca YAML dan mengembalikan data serta path absolutnya."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File YAML tidak ditemukan: {resolved}")
    with resolved.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Root YAML harus berupa mapping: {resolved}")
    return data, resolved


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve path relatif terhadap root proyek atau base yang diberikan."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return ((base or PROJECT_ROOT) / candidate).resolve()


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load an experiment with optional relative `extends` and deep overrides."""
    def merge(base: dict, overrides: dict) -> dict:
        result = dict(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    def read(current: Path, visited: set[Path]) -> dict:
        if current in visited:
            raise ValueError(f"Experiment inheritance cycle: {current}")
        config, resolved = load_yaml(current)
        parent = config.pop("extends", None)
        if parent is None:
            return config
        if not isinstance(parent, str):
            raise ValueError("extends must be a relative YAML path")
        parent_path = resolve_path(parent, resolved.parent)
        return merge(read(parent_path, visited | {resolved}), config)

    resolved = resolve_path(path)
    return read(resolved, set()), resolved


def resolve_model_reference(reference: str | Path) -> str:
    """Gunakan file model lokal jika ada; selain itu pertahankan nama model Ultralytics."""
    candidate = resolve_path(reference)
    return str(candidate) if candidate.exists() else str(reference)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Menghitung SHA-256 file tanpa memuat seluruh file ke memori."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_sha256(data_path: str | Path) -> str:
    """Fingerprint directory-based YOLO splits without modifying dataset files.

    Include split membership, relative filenames, image bytes, labels and mapping.
    Cache files are excluded. This supports legacy data without a source manifest.
    """
    config, resolved = load_yaml(data_path)
    root = resolve_path(config.get("path", "."), resolved.parent)
    digest = hashlib.sha256()
    digest.update(json.dumps({"nc": config.get("nc"), "names": config.get("names")},
                             sort_keys=True, ensure_ascii=False).encode("utf-8"))
    image_extensions = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif",
                        ".tiff", ".webp", ".pfm", ".heic"}
    for split in ("train", "val", "test"):
        if not config.get(split):
            continue
        images = resolve_path(config[split], root)
        if not images.is_dir() or images.name != "images":
            raise ValueError(f"Dataset fingerprint requires a split images directory: {images}")
        labels = images.with_name("labels")
        if not labels.is_dir():
            raise FileNotFoundError(f"Dataset labels directory missing: {labels}")
        image_files = sorted(p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions)
        if not image_files:
            raise ValueError(f"Empty dataset split: {images}")
        label_files = sorted(labels.rglob("*.txt"))
        for directory, files in ((images, image_files), (labels, label_files)):
            for path in files:
                record = [split, directory.name, path.relative_to(directory).as_posix(), sha256_file(path)]
                digest.update((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    """Versi dependency utama yang memengaruhi reproducibility."""
    packages = (
        "torch",
        "torchvision",
        "ultralytics",
        "ultralytics-thop",
        "opencv-python",
        "numpy",
        "PyYAML",
        "psutil",
        "pandas",
        "matplotlib",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def environment_metadata() -> dict[str, Any]:
    """Metadata runtime, OS, CUDA, dan GPU."""
    import torch

    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_gib": round(props.total_memory / (1024**3), 3),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu": gpu,
    }


def json_safe(value: Any) -> Any:
    """Mengonversi object umum dari NumPy/PyTorch/Path agar serializable."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Menulis JSON UTF-8 dengan direktori parent otomatis."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target
