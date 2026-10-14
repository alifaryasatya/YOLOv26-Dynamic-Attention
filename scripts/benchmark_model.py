"""Benchmark parameter, FLOPs, latency forward, dan peak VRAM model YOLO."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params
from attention_runtime import model_gflops

from common import environment_metadata, resolve_model_reference, resolve_path, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile model YOLO26")
    parser.add_argument("--model", default="yolo26n.pt", help="Model/checkpoint .pt atau .yaml")
    parser.add_argument("--device", default="0", help="0 atau cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/profile/baseline_profile.json")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia; gunakan --device cpu")
    index = int(value.split(",")[0])
    return torch.device(f"cuda:{index}")


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.batch <= 0:
        raise ValueError("iterations dan batch harus positif; warmup tidak boleh negatif")

    reference = resolve_model_reference(args.model)
    source_path = resolve_path(args.model)
    device = select_device(args.device)
    use_fp16 = args.fp16 and device.type == "cuda"

    yolo = YOLO(reference)
    network = yolo.model.to(device).eval()
    gflops = model_gflops(network, args.imgsz)
    if use_fp16:
        network.half()
    dtype = torch.float16 if use_fp16 else torch.float32
    sample = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            network(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        durations: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            network(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - start) * 1000.0)

    durations_sorted = sorted(durations)
    p95_index = min(len(durations_sorted) - 1, int(0.95 * len(durations_sorted)))
    mean_batch_ms = statistics.fmean(durations)
    mean_image_ms = mean_batch_ms / args.batch
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None

    result = {
        "model": reference,
        "model_sha256": sha256_file(source_path) if source_path.is_file() else None,
        "device": str(device),
        "dtype": str(dtype),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "parameters": get_num_params(network),
        "trainable_parameters": sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad),
        "gflops": gflops,
        "checkpoint_size_mb": source_path.stat().st_size / (1024**2) if source_path.is_file() else None,
        "forward_latency_ms": {
            "mean_per_batch": mean_batch_ms,
            "mean_per_image": mean_image_ms,
            "median_per_batch": statistics.median(durations),
            "p95_per_batch": durations_sorted[p95_index],
            "fps_from_mean_forward": 1000.0 / mean_image_ms if mean_image_ms > 0 else None,
        },
        "peak_memory_mb": {
            "allocated": peak_allocated / (1024**2) if peak_allocated is not None else None,
            "reserved": peak_reserved / (1024**2) if peak_reserved is not None else None,
        },
        "scope_note": "Pure model forward; tidak mencakup image decode, resize, NMS, atau file I/O.",
        "environment": environment_metadata(),
    }
    output = write_json(resolve_path(args.output), result)
    print("=" * 72)
    print(f"Parameters       : {result['parameters']:,}")
    print(f"GFLOPs           : {gflops:.3f}")
    print(f"Mean forward     : {mean_image_ms:.3f} ms/image")
    print(f"Forward FPS      : {result['forward_latency_ms']['fps_from_mean_forward']:.2f}")
    if peak_allocated is not None:
        print(f"Peak VRAM alloc. : {result['peak_memory_mb']['allocated']:.2f} MB")
    print(f"Report           : {output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
