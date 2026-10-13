"""Evaluate raw-dB validation targets, without optimizer updates."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lunar_rm.config import resolve_config
from lunar_rm.checkpoint import read_checkpoint
from lunar_rm.evaluation import evaluate_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--experiment")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-per-band", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        # With no explicit config, reconstruct the checkpoint's resolved model.
        values = read_checkpoint(args.checkpoint)["config"].copy() if args.checkpoint and not args.config else {}
        values.update({k: getattr(args, k) for k in ("data_root", "device", "batch_size", "num_workers")
                       if getattr(args, k) is not None})
        values["resume"] = None  # evaluation is not a training resume operation
        cfg = resolve_config(args.config, experiment=args.experiment, overrides=values)
        output = args.output.resolve()
        if output.is_relative_to(Path(cfg.data_root)):
            raise ValueError("evaluation output must be outside the read-only data root")
        if output.exists():
            raise ValueError("evaluation output already exists")
        report = evaluate_checkpoint(cfg, args.checkpoint, max_per_band=args.max_per_band)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report["metrics"], allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
