"""Comparison correctness/provenance tests using synthetic report data only."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_phase4 import validate_protocol, compare_records


def manifests():
    base = {
        "status": "completed", "completed_epochs": 15, "configured_epochs": 15,
        "train_arguments": {"epochs": 15, "batch": 16, "seed": 0, "name": "baseline", "project": "runs/phase4"},
        "model_source_sha256": "weights", "dataset_manifest_sha256": "dataset",
        "data_sha256": "yaml", "implementation_sha256": {"module.py": "hash"},
        "environment": {"packages": {"torch": "pinned"}, "python": "pinned"}, "attention": None,
    }
    result = {label: deepcopy(base) for label in ("baseline", "bra_p4", "cbam_p4")}
    for label in ("bra_p4", "cbam_p4"):
        result[label]["attention"] = {"kind": label.replace("_p4", ""), "layer": 19}
        result[label]["train_arguments"]["name"] = label
    return result


def records():
    base = {
        "evaluation": {"data_sha256": "data", "split": "val", "weights_sha256": "best",
                       "arguments": {"imgsz": 640, "batch": 16, "device": "0"},
                       "metrics": {"precision": .5, "recall": .6, "map50": .5, "map50_95": .3}},
        "profile": {"model_sha256": "best", "imgsz": 640, "batch": 1, "device": "cuda:0",
                    "dtype": "float16", "warmup_iterations": 20, "measured_iterations": 100,
                    "environment": {"gpu": {"name": "same GPU"}}, "parameters": 1000,
                    "gflops": 5.0, "forward_latency_ms": {"mean_per_image": 10.0},
                    "peak_memory_mb": {"allocated": 50}},
    }
    result = {label: deepcopy(base) for label in ("baseline", "bra_p4", "cbam_p4")}
    result["bra_p4"]["evaluation"]["metrics"]["map50_95"] = .32
    result["bra_p4"]["profile"]["parameters"] = 1100
    result["bra_p4"]["profile"]["forward_latency_ms"]["mean_per_image"] = 12.0
    return result


class ComparisonTests(unittest.TestCase):
    def test_matches_three_recipes(self):
        validate_protocol(manifests())

    def test_rejects_interim_scheduler_batch_source_and_code_mismatch(self):
        for field, value in (("configured_epochs", 50), ("model_source_sha256", "other"),
                             ("implementation_sha256", {"module.py": "modified"})):
            data = manifests()
            data["bra_p4"][field] = value
            with self.assertRaises(ValueError):
                validate_protocol(data)
        data = manifests()
        data["cbam_p4"]["train_arguments"]["batch"] = 8
        with self.assertRaises(ValueError):
            validate_protocol(data)

    def test_delta_and_percent_computations(self):
        rows = compare_records(records())
        self.assertEqual(rows[0]["delta_map50_95"], 0)
        self.assertAlmostEqual(rows[1]["delta_map50_95"], .02)
        self.assertAlmostEqual(rows[1]["parameter_increase_pct"], 10)
        self.assertAlmostEqual(rows[1]["latency_increase_pct"], 20)

    def test_rejects_test_split_weight_mixup_and_profile_mismatch(self):
        for change in ("test", "weights", "dtype"):
            data = records()
            if change == "test":
                data["cbam_p4"]["evaluation"]["split"] = "test"
            elif change == "weights":
                data["cbam_p4"]["profile"]["model_sha256"] = "wrong"
            else:
                data["cbam_p4"]["profile"]["dtype"] = "float32"
            with self.assertRaises(ValueError):
                compare_records(data)


if __name__ == "__main__":
    unittest.main()
