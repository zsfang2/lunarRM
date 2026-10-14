"""Read one sample per band and check full-resolution inference, without updates."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy  # Initialize NumPy/MKL before PyTorch loads libgomp in Conda environments.
import torch

from lunar_rm.config import resolve_config, validate_runtime
from lunar_rm.data import dataset_from_config
from lunar_rm.models import build_model, parameter_report
from lunar_rm.provenance import code_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments", nargs="+", default=["A0", "N0", "N1", "R0", "G0", "R1", "R2"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    code = code_identity()
    for name in args.experiments:
        cfg = resolve_config(experiment=name, overrides={"data_root": args.data_root, "device": args.device})
        validate_runtime(cfg)
        ds = dataset_from_config(cfg, "val")
        model = build_model(cfg).to(cfg.device).eval()
        for index in (0, len(ds.rows)):
            sample = ds[index]
            baseline = sample.get("baseline_db")
            with torch.inference_mode():
                output = model(sample["features"][None].to(cfg.device),
                               baseline[None].to(cfg.device) if baseline is not None else None)
            if tuple(output.shape) != (1, 1, 256, 256) or not torch.isfinite(output).all():
                raise RuntimeError(f"invalid output for {name}/{sample['band']}")
            print(json.dumps({"experiment": name, "sample_id": sample["sample_id"], "band": sample["band"],
                              "shape": list(output.shape), **parameter_report(model), "training_started": False,
                              "weights": "analytic" if cfg.model == "fspl" else "untrained",
                              "config": cfg.to_dict(), "data_identity": {"val": ds.identity},
                              "code_identity": code}))


if __name__ == "__main__":
    main()
