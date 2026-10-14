"""Hitung metrik sekunder untuk meninjau kandidat near-duplicate dari dataset audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from common import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review kandidat near-duplicate dari laporan audit")
    parser.add_argument("--audit", default="artifacts/audit_v2/dataset_audit.json")
    parser.add_argument("--output", default="artifacts/audit_v2/near_duplicate_review.csv")
    return parser.parse_args()


def label_classes(image_path: Path) -> str:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return ""
    classes = set()
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            classes.add(int(float(line.split()[0])))
    return ",".join(str(value) for value in sorted(classes))


def comparison_metrics(first_path: Path, second_path: Path) -> dict[str, float | int]:
    first = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(second_path), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        raise ValueError(f"Gagal membaca pasangan: {first_path}, {second_path}")

    size = (256, 256)
    first_resized = cv2.resize(first, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    second_resized = cv2.resize(second, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    difference = first_resized - second_resized
    normalized_rmse = float(np.sqrt(np.mean(difference**2)) / 255.0)
    pixel_correlation = float(np.corrcoef(first_resized.ravel(), second_resized.ravel())[0, 1])

    first_hist = cv2.calcHist([first], [0], None, [256], [0, 256])
    second_hist = cv2.calcHist([second], [0], None, [256], [0, 256])
    histogram_correlation = float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CORREL))

    orb = cv2.ORB_create(nfeatures=500)
    first_keypoints, first_descriptors = orb.detectAndCompute(first, None)
    second_keypoints, second_descriptors = orb.detectAndCompute(second, None)
    good_matches = 0
    if first_descriptors is not None and second_descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for candidates in matcher.knnMatch(first_descriptors, second_descriptors, k=2):
            if len(candidates) == 2 and candidates[0].distance < 0.75 * candidates[1].distance:
                good_matches += 1

    return {
        "normalized_rmse": normalized_rmse,
        "pixel_correlation": pixel_correlation,
        "histogram_correlation": histogram_correlation,
        "first_keypoints": len(first_keypoints),
        "second_keypoints": len(second_keypoints),
        "good_orb_matches": good_matches,
    }


def main() -> int:
    args = parse_args()
    audit_path = resolve_path(args.audit)
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    warning = next(
        (item for item in report.get("warnings", []) if item.get("type") == "near_duplicate_candidates"), None
    )
    pairs = warning.get("pairs", []) if warning else []
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "first_split",
        "first_file",
        "first_classes",
        "second_split",
        "second_file",
        "second_classes",
        "phash_distance",
        "normalized_rmse",
        "pixel_correlation",
        "histogram_correlation",
        "first_keypoints",
        "second_keypoints",
        "good_orb_matches",
        "high_similarity_flag",
    ]
    rows = []
    for pair in pairs:
        first_path = Path(pair["first_file"])
        second_path = Path(pair["second_file"])
        metrics = comparison_metrics(first_path, second_path)
        high_similarity = metrics["pixel_correlation"] >= 0.98 and metrics["normalized_rmse"] <= 0.08
        rows.append(
            {
                "first_split": pair["first_split"],
                "first_file": str(first_path),
                "first_classes": label_classes(first_path),
                "second_split": pair["second_split"],
                "second_file": str(second_path),
                "second_classes": label_classes(second_path),
                "phash_distance": pair["distance"],
                **metrics,
                "high_similarity_flag": high_similarity,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reviewed pairs      : {len(rows)}")
    print(f"High-similarity flag: {sum(bool(row['high_similarity_flag']) for row in rows)}")
    print(f"Report              : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
