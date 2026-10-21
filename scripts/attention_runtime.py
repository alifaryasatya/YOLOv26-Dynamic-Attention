"""Make repo-local custom checkpoint classes importable for all entry points."""

import sys
from common import PROJECT_ROOT

source_path = str(PROJECT_ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)

from plantdoc_attention.integrations.yolo26 import (BRAConfig, AttentionDetectionTrainer,
                                                   BRADetectionTrainer, attach_attention_p4, attach_bra_p4)
from plantdoc_attention.integrations.profiling import model_gflops

__all__ = ["BRAConfig", "AttentionDetectionTrainer", "BRADetectionTrainer",
           "attach_attention_p4", "attach_bra_p4", "model_gflops"]
