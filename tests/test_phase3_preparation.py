"""Preparation checks only: no model forward/backward or training."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_experiment_config
from review_phase3_run import assess
from training_telemetry import TrainingTelemetry


class PreparationTests(unittest.TestCase):
    def test_stage_configs_inherit_attention_and_freeze_epochs(self):
        for name, epochs, batch in (("overfit", 50, 4), ("smoke", 1, 16), ("screening", 5, 16)):
            config, _ = load_experiment_config(f"configs/experiments/bra_p4_{name}.yaml")
            self.assertEqual(config["train"]["epochs"], epochs)
            self.assertEqual(config["train"]["batch"], batch)
            self.assertEqual(config["attention"]["gamma_init"], 0)
            self.assertEqual(config["model"], "yolo26n.pt")
            self.assertEqual(config["train"]["optimizer"], "AdamW")
        overfit, _ = load_experiment_config("configs/experiments/bra_p4_overfit.yaml")
        for key in ("mosaic", "mixup", "hsv_h", "translate", "scale", "fliplr", "erasing"):
            self.assertEqual(overfit["train"][key], 0)

    def test_subset_is_train_only_and_preserves_original_labels(self):
        manifest = json.loads((ROOT / "artifacts/phase3_legacy/overfit/subset_manifest.json").read_text())
        self.assertEqual(manifest["images"], 12)
        self.assertEqual(len({item["image"] for item in manifest["selected"]}), 12)
        for item in manifest["selected"]:
            self.assertTrue(Path(item["image"]).is_relative_to(ROOT / "dataset/train/images"))
            self.assertTrue(Path(item["label"]).is_relative_to(ROOT / "dataset/train/labels"))
        import yaml
        subset = yaml.safe_load((ROOT / "artifacts/phase3_legacy/overfit/data.yaml").read_text())
        self.assertEqual(subset["train"], subset["val"])
        self.assertNotIn("test", subset)

    def test_inheritance_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.yaml"
            path.write_text("extends: cycle.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_assessment_rejects_nan_missing_epochs_and_telemetry(self):
        rows = [{"epoch": "1", "train/box_loss": "nan", "train/cls_loss": "1"}]
        report = assess(rows, [], "smoke", 2)
        self.assertEqual(report["status"], "failed")
        self.assertGreaterEqual(len(report["issues"]), 3)

    def test_assessment_accepts_finite_declining_losses(self):
        rows = [{"epoch": str(index), "train/box_loss": str(6 - index)} for index in range(1, 6)]
        telemetry = [{"epoch": index, "gamma": .01, "peak_allocated_mib": 3000,
                      "gpu_total_mib": 6140} for index in range(1, 6)]
        self.assertEqual(assess(rows, telemetry, "screening", 5)["status"], "passed")
        self.assertEqual(assess(rows, telemetry, "overfit", 5)["status"], "passed")
        telemetry[-1]["peak_allocated_mib"] = 5700
        self.assertEqual(assess(rows, telemetry, "screening", 5)["status"], "needs_review")

    def test_telemetry_skips_duplicate_and_final_validation_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "results.csv").write_text("epoch,train/box_loss\n1,1\n", encoding="utf-8")
            network = SimpleNamespace(model=[None] * 19 + [SimpleNamespace(
                residual=SimpleNamespace(gamma=torch.tensor(.1)))])
            trainer = SimpleNamespace(epoch=0, save_dir=path, device=torch.device("cpu"), model=network)
            telemetry = TrainingTelemetry()
            with patch("training_telemetry.unwrap_model", return_value=network):
                telemetry.record_epoch(trainer)
                telemetry.record_epoch(trainer)
                trainer.epoch = 1
                telemetry.record_epoch(trainer)
            self.assertEqual(len(telemetry.epochs), 1)
            self.assertEqual(telemetry.epochs[0]["epoch"], 1)
            self.assertIsNone(telemetry.epochs[0]["peak_allocated_mib"])


if __name__ == "__main__":
    unittest.main()
