"""Epoch callbacks that record gate and GPU peaks without extra forward passes."""

import csv
from pathlib import Path

import torch
from ultralytics.utils.torch_utils import unwrap_model

from common import write_json


class TrainingTelemetry:
    def __init__(self):
        self.epochs = []

    def start_epoch(self, trainer):
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def record_epoch(self, trainer):
        # final_eval fires on_fit_epoch_end with an incremented epoch. Match
        # against results.csv so it cannot create a phantom training epoch.
        csv_path = Path(trainer.save_dir) / "results.csv"
        if not csv_path.is_file():
            return
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epoch = trainer.epoch + 1
        if not rows or int(float(rows[-1]["epoch"])) != epoch:
            return
        if self.epochs and self.epochs[-1]["epoch"] == epoch:
            return
        network = unwrap_model(trainer.model)
        layer = network.model[19]
        gamma = float(layer.residual.gamma.detach()) if hasattr(layer, "residual") else None
        cuda = trainer.device.type == "cuda"
        self.epochs.append({
            "epoch": epoch, "gamma": gamma,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(trainer.device) / 1024**2 if cuda else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(trainer.device) / 1024**2 if cuda else None,
            "gpu_total_mib": torch.cuda.get_device_properties(trainer.device).total_memory / 1024**2 if cuda else None,
        })
        write_json(Path(trainer.save_dir) / "training_telemetry.json", {"epochs": self.epochs})
