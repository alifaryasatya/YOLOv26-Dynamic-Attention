"""Build and save a YOLO26n+BRA-P4 checkpoint for integration checks."""

import argparse
from dataclasses import asdict

from ultralytics import YOLO

from attention_runtime import BRAConfig, attach_attention_p4, model_gflops
from common import environment_metadata, resolve_path, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/phase4_legacy_50e/baseline_50e_seed0/weights/best.pt")
    parser.add_argument("--output")
    args = parser.parse_args()
    default_output = "artifacts/integration_legacy/yolo26n_bra_p4_init.pt"
    source, output = resolve_path(args.weights), resolve_path(args.output or default_output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    config = BRAConfig()
    yolo = YOLO(str(source))
    attach_attention_p4(yolo.model, config)
    parameters = sum(p.numel() for p in yolo.model.parameters())
    gflops = model_gflops(yolo.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Original optimizer state does not match newly inserted parameters. This
    # initialized artifact is for integration/inference, not training resume.
    yolo.ckpt = {
        "epoch": -1, "optimizer": None, "best_fitness": None,
        "train_args": {**yolo.model.args, "model": str(output)},
    }
    yolo.save(str(output))
    write_json(output.with_suffix(".json"), {
        "status": "initialized_untrained_attention",
        "source": str(source), "source_sha256": sha256_file(source),
        "output": str(output), "output_sha256": sha256_file(output),
        "attention": asdict(config), "parameters": parameters,
        "gflops_estimate": gflops,
        "environment": environment_metadata(),
        "note": "Integration checkpoint from legacy baseline; ablation config starts from yolo26n.pt.",
    })
    print(f"Created {output} | parameters={parameters:,} | GFLOPs estimate={gflops:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
