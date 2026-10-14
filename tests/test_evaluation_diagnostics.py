"""Diagnostic validator controls restore state, including on failures."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_legacy_evaluation import AMPValidator, RecordedValidator, DetectionValidator, nms


class DiagnosticTests(unittest.TestCase):
    def test_nms_deadline_disabled_and_function_restored(self):
        validator = RecordedValidator.__new__(RecordedValidator)
        validator.disable_nms_time_limit = True
        validator.simulated_stop_after = None
        validator.batch_i = 0
        outputs = [{"bboxes": torch.ones(2, 4)}]
        with patch.object(nms, "non_max_suppression", return_value=outputs) as original:
            with patch.object(DetectionValidator, "postprocess",
                              side_effect=lambda _: nms.non_max_suppression(None)):
                self.assertIs(validator.postprocess(None), outputs)
            original.assert_called_once_with(None, max_time_img=float("inf"))
            self.assertIs(nms.non_max_suppression, original)
            with patch.object(DetectionValidator, "postprocess", side_effect=RuntimeError("failure")):
                with self.assertRaises(RuntimeError):
                    validator.postprocess(None)
            self.assertIs(nms.non_max_suppression, original)

    def test_amp_input_preprocessing_precision_restored(self):
        validator = AMPValidator.__new__(AMPValidator)
        validator.args = SimpleNamespace(quantize=None)
        def preprocess(batch):
            self.assertEqual(validator.args.quantize, 16)
            return batch
        with patch.object(DetectionValidator, "preprocess", side_effect=preprocess):
            self.assertEqual(validator.preprocess({"test": 1}), {"test": 1})
        self.assertIsNone(validator.args.quantize)
        with patch.object(DetectionValidator, "preprocess", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                validator.preprocess({})
        self.assertIsNone(validator.args.quantize)

    def test_timeout_simulation_only_discards_marked_first_batch_tail(self):
        validator = RecordedValidator.__new__(RecordedValidator)
        validator.disable_nms_time_limit = False
        validator.simulated_stop_after = 1
        validator.batch_i = 0
        outputs = [{"bboxes": torch.ones(2, 4)} for _ in range(3)]
        with patch.object(DetectionValidator, "postprocess", return_value=outputs):
            result = validator.postprocess(None)
        self.assertEqual([len(item["bboxes"]) for item in result], [2, 0, 0])
        validator.batch_i = 1
        outputs = [{"bboxes": torch.ones(2, 4)} for _ in range(3)]
        with patch.object(DetectionValidator, "postprocess", return_value=outputs):
            self.assertEqual([len(item["bboxes"]) for item in validator.postprocess(None)], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
