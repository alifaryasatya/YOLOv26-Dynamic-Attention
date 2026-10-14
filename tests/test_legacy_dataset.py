"""Legacy suite configuration and content provenance, without training."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import dataset_content_sha256, load_experiment_config, write_json
from compare_phase4 import make_report, suite_settings, validate_protocol
from test_phase4_comparison import manifests, records


class LegacyDatasetTests(unittest.TestCase):
    def test_matched_fifty_epoch_configs_and_protocol(self):
        baseline, _ = load_experiment_config("configs/experiments/legacy_baseline_50e.yaml")
        bra, _ = load_experiment_config("configs/experiments/legacy_bra_p4_50e.yaml")
        self.assertEqual(baseline["train"], bra["train"])
        self.assertEqual(bra["train"]["epochs"], 50)
        self.assertEqual(bra["train"]["patience"], 0)
        self.assertEqual(bra["train"]["close_mosaic"], 10)
        self.assertEqual(baseline["data"], bra["data"])
        self.assertEqual(bra["data"], "plantdoc_data.yaml")
        self.assertFalse(baseline.get("attention"))
        self.assertEqual(bra["attention"]["kind"], "bra")
        suite = suite_settings("legacy", 50)
        for label, config in (("baseline", baseline), ("bra_p4", bra)):
            self.assertEqual(config["experiment"]["project"], suite["run_root"])
            self.assertEqual(config["experiment"]["name"], suite["models"][label])
        data = {k: v for k, v in manifests().items() if k != "cbam_p4"}
        for manifest in data.values():
            manifest.update(completed_epochs=50, configured_epochs=50)
            manifest["train_arguments"]["epochs"] = 50
        validate_protocol(data, epochs=50)
        data["bra_p4"]["completed_epochs"] = 15
        with self.assertRaises(ValueError):
            validate_protocol(data, epochs=50)
        with self.assertRaises(ValueError):
            suite_settings("unknown", 50)

    def test_matched_recipe_and_separate_outputs(self):
        baseline, _ = load_experiment_config("configs/experiments/legacy_baseline_15e.yaml")
        bra, _ = load_experiment_config("configs/experiments/legacy_bra_p4_15e.yaml")
        self.assertEqual(baseline["data"], "plantdoc_data.yaml")
        self.assertEqual(baseline["data"], bra["data"])
        self.assertEqual(baseline["model"], bra["model"])
        self.assertEqual(baseline["train"], bra["train"])
        self.assertEqual(baseline["train"]["epochs"], 15)
        self.assertFalse(baseline.get("attention"))
        self.assertEqual(bra["attention"]["kind"], "bra")
        for config in (baseline, bra):
            self.assertEqual(config["experiment"]["project"], "runs/phase4_legacy")
        self.assertEqual(list(suite_settings("legacy", 15)["models"]), ["baseline", "bra_p4"])
        self.assertEqual(suite_settings()["run_root"], "runs/phase4_legacy_50e")

    def test_content_provenance_replaces_missing_source_manifest(self):
        data = {k: v for k, v in manifests().items() if k != "cbam_p4"}
        for manifest in data.values():
            manifest["dataset_manifest_sha256"] = None
            manifest["dataset_content_sha256"] = "same files"
        validate_protocol(data)
        mismatch = deepcopy(data)
        mismatch["bra_p4"]["dataset_content_sha256"] = "different files"
        with self.assertRaises(ValueError):
            validate_protocol(mismatch)
        del data["bra_p4"]["dataset_content_sha256"]
        with self.assertRaises(ValueError):
            validate_protocol(data)

    def test_fingerprint_detects_image_label_mapping_and_split_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yaml = root / "data.yaml"
            yaml.write_text("path: .\ntrain: train/images\nval: valid/images\nnc: 1\nnames: [leaf]\n", encoding="utf-8")
            for split in ("train", "valid"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir()
                (root / split / "images/a.jpg").write_bytes(b"image fixture")
                (root / split / "labels/a.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
            original = dataset_content_sha256(yaml)
            self.assertEqual(original, dataset_content_sha256(yaml))
            (root / "train/labels.cache").write_bytes(b"ignore cache")
            self.assertEqual(original, dataset_content_sha256(yaml))
            for target, content in ((root / "train/images/a.jpg", b"other image"),
                                    (root / "train/labels/a.txt", b"0 .5 .5 .3 .3\n"),
                                    (yaml, yaml.read_bytes().replace(b"[leaf]", b"[disease]"))):
                before = target.read_bytes()
                target.write_bytes(content)
                self.assertNotEqual(original, dataset_content_sha256(yaml))
                target.write_bytes(before)
            (root / "train/images/a.jpg").rename(root / "train/images/b.jpg")
            self.assertNotEqual(original, dataset_content_sha256(yaml))

    def test_two_model_report_does_not_require_cbam(self):
        suite = suite_settings("legacy", 15)
        fixtures = records()
        matched = {label: {"best_checkpoint_sha256": "best", "data_sha256": "data"}
                   for label in suite["models"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, evaluations, artifacts = root / "runs", root / "eval", root / "artifacts"
            for label, name in suite["models"].items():
                evaluation = evaluations / f"legacy_{label}_val"
                write_json(evaluation / "metrics_summary.json", fixtures[label]["evaluation"])
                (evaluation / "per_class_metrics.csv").write_text(
                    "class_id,class_name,map50_95\n0,leaf,0.3\n", encoding="utf-8")
                write_json(artifacts / "profiles" / f"legacy_{label}.json", fixtures[label]["profile"])
                write_json(runs / name / "training_telemetry.json", {"epochs": [{"peak_allocated_mib": 100}]})
            with patch("compare_phase4.inspect_runs", return_value=matched) as inspect:
                output = make_report(runs, evaluations, artifacts, "legacy",
                                     data=suite["data"], models=suite["models"])
                inspect.assert_called_once_with(runs, data=suite["data"], models=suite["models"], epochs=15)
            self.assertTrue((output / "COMPARISON.md").is_file())
            self.assertIn("bra_p4_delta", (output / "per_class_comparison.csv").read_text())
            self.assertNotIn("cbam", (output / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
