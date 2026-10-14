"""Evaluate/profile completed legacy-dataset Phase 4 runs and compare."""

import argparse
import subprocess
import sys

from common import PROJECT_ROOT, resolve_path
from compare_phase4 import inspect_runs, make_report, suite_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dataset", choices=("legacy",), default="legacy")
    parser.add_argument("--epochs", type=int, choices=(15, 50), default=50)
    parser.add_argument("--tag", help="Unique output tag for evaluation/profile")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without model evaluation or profiling")
    args = parser.parse_args()
    suite = suite_settings(args.dataset, args.epochs)
    args.tag = args.tag or suite["tag"]
    run_root = resolve_path(suite["run_root"])
    evaluation_root, artifact_root = resolve_path("runs/evaluation"), resolve_path(suite["artifact_root"])
    if not args.dry_run:
        inspect_runs(run_root, data=suite["data"], models=suite["models"], epochs=args.epochs)
    commands = []
    for label, name in suite["models"].items():
        run = run_root / name
        weights = run / "weights/best.pt"
        evaluation_name = f"{args.tag}_{label}_val"
        profile = artifact_root / "profiles" / f"{args.tag}_{label}.json"
        if (evaluation_root / evaluation_name).exists() or profile.exists():
            raise FileExistsError("Output already exists; choose a new --tag to retain prior results")
        commands += [
            [sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"), "--weights", str(weights),
             "--data", str(resolve_path(suite["data"])), "--split", "val", "--imgsz", "640",
             "--batch", "16", "--device", args.device, "--project", str(evaluation_root), "--name", evaluation_name],
            [sys.executable, str(PROJECT_ROOT / "scripts/benchmark_model.py"), "--model", str(weights),
             "--imgsz", "640", "--batch", "1", "--device", args.device, "--warmup", "20",
             "--iterations", "100", "--fp16", "--output", str(profile)],
        ]
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not args.dry_run:
        output = make_report(run_root, evaluation_root, artifact_root, args.tag,
                             data=suite["data"], models=suite["models"], epochs=args.epochs)
        print(f"Comparison saved: {output}")
    else:
        print("DRY RUN: no training, evaluation, or profiling started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
