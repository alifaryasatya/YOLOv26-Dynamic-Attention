"""Audit integritas, distribusi kelas, dan potensi leakage dataset YOLO."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PROJECT_ROOT, load_yaml, resolve_path, sha256_file, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_KEYS = ("train", "val", "test")
MAX_REPORTED_PAIRS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset PlantDoc berformat YOLO")
    parser.add_argument("--data", default="plantdoc_data.yaml", help="Path konfigurasi dataset")
    parser.add_argument("--output", default="artifacts/audit_v2", help="Direktori laporan")
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="Maksimum Hamming distance untuk kandidat near-duplicate (0 menonaktifkan)",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero jika ditemukan error/leakage eksak")
    return parser.parse_args()


def dataset_root(config: dict[str, Any], config_path: Path) -> Path:
    root = Path(config.get("path", "."))
    if root.is_absolute():
        return root.resolve()
    return (config_path.parent / root).resolve()


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def perceptual_hash(path: Path) -> int | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct.flatten()[1:]
    bits = values > np.median(values)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def audit_label(
    path: Path,
    nc: int,
    split: str,
    class_counts: dict[str, Counter[int]],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> int:
    object_count = 0
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        errors.append({"type": "label_encoding", "file": str(path), "message": str(exc)})
        return 0

    if not lines:
        warnings.append({"type": "empty_label", "file": str(path)})
        return 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(
                {
                    "type": "invalid_field_count",
                    "file": str(path),
                    "line": line_number,
                    "value": raw_line,
                }
            )
            continue
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append({"type": "non_numeric_label", "file": str(path), "line": line_number})
            continue

        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < nc:
            errors.append(
                {
                    "type": "invalid_class_id",
                    "file": str(path),
                    "line": line_number,
                    "value": fields[0],
                }
            )
            continue
        if not all(math.isfinite(value) for value in coordinates):
            errors.append({"type": "non_finite_box", "file": str(path), "line": line_number})
            continue

        x_center, y_center, width, height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates) or width <= 0.0 or height <= 0.0:
            errors.append(
                {
                    "type": "invalid_box_range",
                    "file": str(path),
                    "line": line_number,
                    "values": coordinates,
                }
            )
            continue

        tolerance = 1e-4
        if (
            x_center - width / 2 < -tolerance
            or y_center - height / 2 < -tolerance
            or x_center + width / 2 > 1 + tolerance
            or y_center + height / 2 > 1 + tolerance
        ):
            warnings.append(
                {
                    "type": "box_crosses_image_boundary",
                    "file": str(path),
                    "line": line_number,
                    "values": coordinates,
                }
            )

        class_counts[split][class_id] += 1
        object_count += 1
    return object_count


def find_cross_split_duplicates(
    records: list[dict[str, Any]], phash_threshold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)

    exact: list[dict[str, Any]] = []
    for digest, members in by_sha.items():
        splits = {member["split"] for member in members}
        if len(splits) > 1:
            exact.append(
                {
                    "sha256": digest,
                    "splits": sorted(splits),
                    "files": [member["path"] for member in members],
                }
            )

    if phash_threshold <= 0:
        return exact, [], 0

    near: list[dict[str, Any]] = []
    near_count = 0
    split_records = {split: [record for record in records if record["split"] == split] for split in SPLIT_KEYS}
    for first_index, first_split in enumerate(SPLIT_KEYS):
        for second_split in SPLIT_KEYS[first_index + 1 :]:
            for first in split_records[first_split]:
                if first["phash"] is None:
                    continue
                for second in split_records[second_split]:
                    if second["phash"] is None or first["sha256"] == second["sha256"]:
                        continue
                    distance = (first["phash"] ^ second["phash"]).bit_count()
                    if distance <= phash_threshold:
                        near_count += 1
                        if len(near) < MAX_REPORTED_PAIRS:
                            near.append(
                                {
                                    "distance": distance,
                                    "first_split": first_split,
                                    "first_file": first["path"],
                                    "second_split": second_split,
                                    "second_file": second["path"],
                                }
                            )
    return exact, near, near_count


def main() -> int:
    args = parse_args()
    config, config_path = load_yaml(args.data)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc = int(config.get("nc", 0))
    names_raw = config.get("names", {})
    if isinstance(names_raw, list):
        names = {index: name for index, name in enumerate(names_raw)}
    else:
        names = {int(index): name for index, name in names_raw.items()}

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if nc <= 0 or len(names) != nc or set(names) != set(range(nc)):
        errors.append(
            {
                "type": "invalid_class_config",
                "nc": nc,
                "name_count": len(names),
                "class_ids": sorted(names),
            }
        )

    root = dataset_root(config, config_path)
    class_counts = {split: Counter() for split in SPLIT_KEYS}
    split_summary: dict[str, Any] = {}
    image_records: list[dict[str, Any]] = []

    for split in SPLIT_KEYS:
        relative_images = config.get(split)
        if not relative_images:
            errors.append({"type": "missing_split_config", "split": split})
            continue
        images_dir = (root / relative_images).resolve()
        labels_dir = images_dir.parent / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(
                {
                    "type": "missing_split_directory",
                    "split": split,
                    "images": str(images_dir),
                    "labels": str(labels_dir),
                }
            )
            continue

        images = image_files(images_dir)
        labels = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        if missing_labels:
            errors.append({"type": "missing_labels", "split": split, "stems": missing_labels})
        if orphan_labels:
            errors.append({"type": "orphan_labels", "split": split, "stems": orphan_labels})

        object_count = 0
        for label in labels:
            object_count += audit_label(label, nc, split, class_counts, errors, warnings)

        unreadable_images = 0
        for index, image in enumerate(images, start=1):
            phash = perceptual_hash(image)
            if phash is None:
                unreadable_images += 1
                errors.append({"type": "unreadable_image", "split": split, "file": str(image)})
                continue
            image_records.append(
                {
                    "split": split,
                    "path": str(image),
                    "sha256": sha256_file(image),
                    "phash": phash,
                }
            )
            if index % 500 == 0:
                print(f"[{split}] hashed {index}/{len(images)} images")

        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": object_count,
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "unreadable_images": unreadable_images,
        }

    exact_duplicates, near_duplicates, near_duplicate_count = find_cross_split_duplicates(
        image_records, args.phash_threshold
    )
    if exact_duplicates:
        errors.append({"type": "exact_cross_split_duplicates", "groups": exact_duplicates})
    if near_duplicate_count:
        warnings.append(
            {
                "type": "near_duplicate_candidates",
                "threshold": args.phash_threshold,
                "count": near_duplicate_count,
                "reported": len(near_duplicates),
                "pairs": near_duplicates,
            }
        )

    distribution_path = output_dir / "class_distribution.csv"
    with distribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", *SPLIT_KEYS, "total"])
        for class_id in range(nc):
            counts = [class_counts[split][class_id] for split in SPLIT_KEYS]
            writer.writerow([class_id, names.get(class_id, ""), *counts, sum(counts)])

    report = {
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(root),
        "nc": nc,
        "names": names,
        "splits": split_summary,
        "totals": {
            "images": sum(item.get("images", 0) for item in split_summary.values()),
            "labels": sum(item.get("labels", 0) for item in split_summary.values()),
            "objects": sum(item.get("objects", 0) for item in split_summary.values()),
        },
        "exact_cross_split_duplicate_groups": len(exact_duplicates),
        "near_duplicate_candidate_pairs": near_duplicate_count,
        "phash_threshold": args.phash_threshold,
        "errors": errors,
        "warnings": warnings,
        "class_distribution_csv": str(distribution_path),
    }
    report_path = write_json(output_dir / "dataset_audit.json", report)

    print("=" * 64)
    print(f"Dataset audit status : {report['status']}")
    print(f"Images / labels      : {report['totals']['images']} / {report['totals']['labels']}")
    print(f"Objects              : {report['totals']['objects']}")
    print(f"Errors / warnings    : {len(errors)} / {len(warnings)}")
    print(f"Exact leakage groups : {len(exact_duplicates)}")
    print(f"Near-duplicate pairs : {near_duplicate_count} (threshold={args.phash_threshold})")
    print(f"Report               : {report_path}")
    print("=" * 64)
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
