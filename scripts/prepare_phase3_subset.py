"""Prepare a deterministic overfit image list from the train split only."""

import argparse
import hashlib
from pathlib import Path

import yaml

from common import load_yaml, resolve_path, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="plantdoc_data.yaml")
    parser.add_argument("--output", default="artifacts/phase3_legacy/overfit")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 4 <= args.count <= 16:
        parser.error("--count must be between 4 and 16")
    data, data_path = load_yaml(args.data)
    root = (data_path.parent / data["path"]).resolve()
    images_dir = root / data["train"]
    labels_dir = images_dir.parent / "labels"
    candidates = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = labels_dir / f"{image.stem}.txt"
        lines = [line for line in label.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        classes = {int(line.split()[0]) for line in lines}
        rank = hashlib.sha256(f"{args.seed}:{image.name}".encode()).hexdigest()
        candidates.append((image, label, classes, len(lines), rank))
    if len(candidates) < args.count:
        raise ValueError("Not enough labeled train images")
    selected, covered = [], set()
    while len(selected) < args.count:
        candidate = min(candidates, key=lambda item: (-len(item[2] - covered), item[3], item[4]))
        candidates.remove(candidate)
        selected.append(candidate)
        covered.update(candidate[2])
    output = resolve_path(args.output)
    list_path, yaml_path = output / "images.txt", output / "data.yaml"
    image_text = "\n".join(str(item[0].resolve()) for item in selected) + "\n"
    subset_config = {"path": str(root), "train": str(list_path), "val": str(list_path),
                     "nc": data["nc"], "names": data["names"]}
    yaml_text = yaml.safe_dump(subset_config, sort_keys=False, allow_unicode=True)
    if output.exists():
        if (list_path.is_file() and yaml_path.is_file() and (output / "subset_manifest.json").is_file()
                and list_path.read_text() == image_text and yaml_path.read_text() == yaml_text):
            print(f"Subset already prepared: {yaml_path}")
            return 0
        raise FileExistsError(f"Output differs from requested subset; choose another --output: {output}")
    output.mkdir(parents=True)
    list_path.write_text(image_text, encoding="utf-8")
    yaml_path.write_text(yaml_text, encoding="utf-8")
    write_json(output / "subset_manifest.json", {
        "source_config": str(data_path), "source_config_sha256": sha256_file(data_path),
        "source_split": "train", "seed": args.seed, "images": len(selected),
        "objects": sum(item[3] for item in selected), "class_ids": sorted(covered),
        "diagnostic_only": True, "train_and_val_identical": True,
        "selected": [{"image": str(image), "image_sha256": sha256_file(image),
                      "label": str(label), "label_sha256": sha256_file(label),
                      "classes": sorted(classes), "objects": objects}
                     for image, label, classes, objects, _ in selected],
    })
    print(f"Prepared {len(selected)} train images, {len(covered)} classes: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
