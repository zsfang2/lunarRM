"""Validate a training configuration; optimization requires --execute."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lunar_rm.config import resolve_config, validate_runtime
from lunar_rm.checkpoint import read_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--experiment")
    for key in ("data-root", "output-dir", "device", "resume", "loss"):
        parser.add_argument(f"--{key}")
    for key in ("epochs", "batch-size", "num-workers", "seed", "width", "groups"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--threads", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="explicitly start optimizer updates")
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    args = vars(parser.parse_args(argv))
    execute, threads = args.pop("execute"), args.pop("threads")
    args.pop("dry_run")
    path, experiment = args.pop("config"), args.pop("experiment")
    try:
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        values = read_checkpoint(args["resume"])["config"].copy() if args["resume"] and not path else {}
        values.update({k: v for k, v in args.items() if v is not None})
        cfg = resolve_config(path, experiment=experiment, overrides=values)
        validate_runtime(cfg, for_training=True)
        if cfg.batch_size % len(cfg.bands):
            raise ValueError("batch_size must be divisible by band count")
        if execute:
            from lunar_rm.engine import train
            train(cfg)
        else:
            print(json.dumps({"training_started": False, "mode": "dry-run", "config": cfg.to_dict()}, indent=2))
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
