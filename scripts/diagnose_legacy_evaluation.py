"""Controlled validation-only experiments: precision, fusion and training AMP."""

import argparse
from functools import partial
from types import MethodType

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import nms
from ultralytics.utils.torch_utils import init_seeds

import attention_runtime
from common import resolve_path, sha256_file, write_json


class RecordedValidator(DetectionValidator):
    simulated_stop_after = None
    disable_nms_time_limit = False

    def postprocess(self, preds):
        original_nms = nms.non_max_suppression
        if self.disable_nms_time_limit:
            nms.non_max_suppression = partial(original_nms, max_time_img=float("inf"))
        try:
            outputs = super().postprocess(preds)
        finally:
            nms.non_max_suppression = original_nms
        if self.batch_i == 0 and self.simulated_stop_after is not None:
            for output in outputs[self.simulated_stop_after:]:
                for key in output:
                    output[key] = output[key][:0]
        return outputs

    def init_metrics(self, model):
        super().init_metrics(model)
        native = model.model
        self.actual = {k: getattr(self.args, k) for k in
                       ("imgsz", "batch", "rect", "conf", "iou", "max_det", "nms",
                        "quantize", "augment", "workers", "seed", "deterministic")}
        self.actual.update(head_end2end=model.end2end,
                           weight_dtype=str(next(native.parameters()).dtype),
                           batchnorm_layers=sum(isinstance(m, torch.nn.BatchNorm2d) for m in native.modules()))
        self.actual["first_batch_files"] = self.dataloader.dataset.im_files[:self.args.batch]


class AMPValidator(RecordedValidator):
    def preprocess(self, batch):
        precision = self.args.quantize
        self.args.quantize = 16
        try:
            return super().preprocess(batch)
        finally:
            self.args.quantize = precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "bra_p4"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "amp"), required=True)
    parser.add_argument("--fusion", choices=("on", "off"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--simulate-first-batch-stop-after", type=int,
                        help="Diagnostic simulation only: discard remaining first-batch predictions like an NMS timeout")
    parser.add_argument("--disable-nms-time-limit", action="store_true",
                        help="Process all validation predictions without the wall-clock NMS cutoff")
    args = parser.parse_args()
    if args.simulate_first_batch_stop_after is not None and not 0 <= args.simulate_first_batch_stop_after < 16:
        parser.error("simulated stop must be in 0..15")
    RecordedValidator.simulated_stop_after = args.simulate_first_batch_stop_after
    RecordedValidator.disable_nms_time_limit = args.disable_nms_time_limit
    name = f"consistency_{args.tag}_{args.model}_{args.precision}_fusion_{args.fusion}"
    output = resolve_path("runs/evaluation") / name
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    run_name = "baseline_50e_seed0" if args.model == "baseline" else "bra_p4_50e_seed0"
    weights = resolve_path("runs/phase4_legacy_50e") / run_name / "weights/best.pt"
    data = resolve_path("plantdoc_data.yaml")
    weights_hash = sha256_file(weights)
    init_seeds(0, deterministic=True)
    yolo = YOLO(str(weights))
    if args.precision == "amp":
        def amp_forward(network, *inputs, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return type(network).forward(network, *inputs, **kwargs)
        yolo.model.forward = MethodType(amp_forward, yolo.model)
    if args.fusion == "off":
        def no_fusion(network, *inputs, **kwargs):
            return network
        yolo.model.fuse = MethodType(no_fusion, yolo.model)
    actual = {}
    def capture(validator):
        actual.update(validator.actual)
    yolo.add_callback("on_val_end", capture)
    recipe = dict(data=str(data), split="val", imgsz=640, batch=16, device="0",
                  workers=2, rect=True, conf=0.001, iou=0.7, max_det=300, nms=True,
                  quantize=16 if args.precision == "fp16" else None, augment=False,
                  seed=0, deterministic=True, plots=False, verbose=False,
                  project=str(output.parent), name=name, exist_ok=False)
    metrics = yolo.val(validator=AMPValidator if args.precision == "amp" else RecordedValidator, **recipe)
    payload = {"model": args.model, "precision": args.precision, "fusion": args.fusion,
               "weights": str(weights), "weights_sha256": weights_hash, "data_sha256": sha256_file(data),
               "requested_arguments": recipe, "actual_arguments": actual,
               "metrics": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                           "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
               "checkpoint_training_metrics": yolo.ckpt.get("train_metrics"),
               "per_class_map50_95": metrics.box.maps.tolist(),
               "class_names": yolo.names,
               "simulated_first_batch_stop_after": args.simulate_first_batch_stop_after,
               "nms_time_limit_disabled": args.disable_nms_time_limit,
               "checkpoint_unchanged": sha256_file(weights) == weights_hash}
    target = write_json(output / "diagnostic_summary.json", payload)
    print(f"DIAGNOSTIC: {payload['metrics']}\nSaved: {target}", flush=True)


if __name__ == "__main__":
    main()
