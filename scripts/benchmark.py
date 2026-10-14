"""Benchmark batch-one inference from raw in-memory DEM, without training."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy  # Initialize NumPy/MKL before PyTorch loads libgomp in Conda environments.
import torch
from lunar_rm.config import resolve_config, validate_runtime
from lunar_rm.checkpoint import read_checkpoint
from lunar_rm.evaluation import prepare_evaluation
from lunar_rm.data import dataset_from_config
from lunar_rm.models import build_model
from lunar_rm.benchmark import benchmark_model
from lunar_rm.provenance import code_identity
from lunar_rm.storage import file_identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--experiment")
    p.add_argument("--checkpoint")
    p.add_argument("--allow-untrained", action="store_true", help="explicitly time random weights; no accuracy claim")
    p.add_argument("--data-root")
    p.add_argument("--device")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(a.threads)
        values = read_checkpoint(a.checkpoint)["config"].copy() if a.checkpoint and not a.config else {}
        values.update({k: getattr(a, k) for k in ("data_root", "device") if getattr(a, k) is not None})
        values["resume"] = None
        cfg = resolve_config(a.config, experiment=a.experiment, overrides=values)
        validate_runtime(cfg)
        output = a.output.resolve()
        if output.exists() or output.is_relative_to(Path(cfg.data_root)):
            raise ValueError("output must be new and outside data root")
        if a.allow_untrained and a.checkpoint:
            raise ValueError("choose checkpoint or allow-untrained, not both")
        if a.allow_untrained:
            torch.manual_seed(cfg.seed)
            model, ds = build_model(cfg), dataset_from_config(cfg, "val")
        else:
            model, ds = prepare_evaluation(cfg, a.checkpoint)
        report = benchmark_model(model, ds, a.index, device=cfg.device, warmup=a.warmup, repeats=a.repeats, amp=cfg.amp)
        report.update(config=cfg.to_dict(), data_identity={"val": ds.identity}, code_identity=code_identity(),
                      checkpoint=file_identity(a.checkpoint, content_hash=True) if a.checkpoint else None,
                      weights="checkpoint" if a.checkpoint else ("analytic" if cfg.model == "fspl" else "untrained"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps(report["timings"], allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
