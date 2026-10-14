"""Custom checkpoint survives serialization and a fresh Python process."""

from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plantdoc_attention.integrations import P4AttentionWrapper, attach_bra_p4
from test_yolo26_integration import assert_outputs_equal


class CheckpointTests(unittest.TestCase):
    def test_reload_nonzero_gate_and_fresh_process_prediction(self):
        yolo = YOLO(str(ROOT / "runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt"))
        attach_bra_p4(yolo.model)
        with torch.no_grad():
            yolo.model.model[19].residual.gamma.fill_(0.125)
        # Model.save serializes FP16 weights; compare against that same precision.
        reference = deepcopy(yolo.model).half().float().eval()
        temp_root = ROOT / "artifacts/tests/bra_checkpoint"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            checkpoint = Path(directory) / "bra.pt"
            yolo.save(str(checkpoint))
            loaded = YOLO(str(checkpoint)).model.float().eval()
            self.assertIsInstance(loaded.model[19], P4AttentionWrapper)
            self.assertEqual(loaded.model[19].residual.gamma.item(), .125)
            self.assertEqual(loaded.yaml["plantdoc_attention"], reference.yaml["plantdoc_attention"])
            sample = torch.rand(1, 3, 128, 128)
            with torch.no_grad():
                assert_outputs_equal(self, reference(sample), loaded(sample), atol=0, rtol=0)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            code = (
                "import sys, torch; from ultralytics import YOLO; "
                "from plantdoc_attention.integrations import P4AttentionWrapper; "
                "m=YOLO(sys.argv[1]); assert isinstance(m.model.model[19],P4AttentionWrapper); "
                "r=m.predict(torch.rand(1,3,128,128),imgsz=128,device='cpu',save=False,verbose=False); "
                "assert len(r)==1; print('fresh_process_prediction=passed')"
            )
            result = subprocess.run([sys.executable, "-c", code, str(checkpoint)],
                                    env=env, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("fresh_process_prediction=passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
