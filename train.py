"""Train a LunarRM model; use --dry-run to check settings without training."""

import argparse
import json
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy  # Initialize NumPy/MKL before PyTorch loads libgomp in Conda environments.
import torch
from training.config import resolve_config, validate_runtime
from training.checkpoint import read_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--experiment")
    for key in ("data-root", "output-dir", "resume", "loss", "cache-dir"):
        parser.add_argument(f"--{key}")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--device")
    device.add_argument("-gpu", "--gpu", type=int, help="visible CUDA index; equivalent to --device cuda:N")
    for key in ("epochs", "batch-size", "num-workers", "seed", "width", "groups"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--threads", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--dry-run", action="store_true", help="validate only; do not train")
    args = vars(parser.parse_args(argv))
    args.pop("execute")
    threads = args.pop("threads")
    execute = not args.pop("dry_run")
    path, experiment = args.pop("config"), args.pop("experiment")
    gpu = args.pop("gpu")
    try:
        if gpu is not None:
            if gpu < 0:
                raise ValueError("gpu index must be nonnegative")
            args["device"] = f"cuda:{gpu}"
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
            from training.engine import train
            train(cfg)
        else:
            print(json.dumps({"training_started": False, "mode": "dry-run", "config": cfg.to_dict()}, indent=2))
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
