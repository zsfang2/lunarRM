"""Grouped validation and shared fixed example panels; never trains."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy  # Initialize NumPy/MKL before PyTorch loads libgomp in Conda environments.
import torch
from lunar_rm.config import resolve_config
from lunar_rm.checkpoint import read_checkpoint
from lunar_rm.evaluation import prepare_evaluation, evaluate_model
from lunar_rm.reporting import fixed_indices, render_examples
from lunar_rm.provenance import code_identity
from lunar_rm.storage import file_identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--experiment")
    p.add_argument("--checkpoint")
    p.add_argument("--data-root")
    p.add_argument("--device")
    p.add_argument("--max-per-band", type=int)
    p.add_argument("--coverage-edges", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--examples", type=Path, required=True, help="shared JSON list of band/sample_id objects")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    try:
        if a.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(a.threads)
        values = read_checkpoint(a.checkpoint)["config"].copy() if a.checkpoint and not a.config else {}
        values.update({k: getattr(a, k) for k in ("data_root", "device") if getattr(a, k) is not None})
        values.update(resume=None, num_workers=0)
        cfg = resolve_config(a.config, experiment=a.experiment, overrides=values)
        out = a.output_dir.resolve()
        if out.exists() or out.is_relative_to(Path(cfg.data_root)) or Path(cfg.data_root).is_relative_to(out):
            raise ValueError("output must be a new directory separate from data")
        identities = json.loads(a.examples.read_text())
        model, ds = prepare_evaluation(cfg, a.checkpoint, need_los=True)
        fixed_indices(ds, identities)
        metrics = evaluate_model(model, ds, device=cfg.device, batch_size=cfg.batch_size,
                                 max_per_band=a.max_per_band, grouped=True, coverage_edges=a.coverage_edges)
        out.mkdir(parents=True, exist_ok=False)
        render_examples(model, ds, identities, out, cfg.device)
        report = {"metrics": metrics, "config": cfg.to_dict(), "examples": identities,
                  "examples_manifest": file_identity(a.examples, content_hash=True),
                  "data_identity": {"val": ds.identity}, "code_identity": code_identity(),
                  "checkpoint": file_identity(a.checkpoint, content_hash=True) if a.checkpoint else None,
                  "training_started": False, "target": "raw_db",
                  "profile": {"step_m": cfg.profile_step_m, "dtype": cfg.feature_dtype}}
        with (out / "diagnostics.json").open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"Wrote diagnostics to {out}")
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
