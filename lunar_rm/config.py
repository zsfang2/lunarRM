"""Resolve and validate experiment settings without starting training.

Precedence: common defaults < experiment profile < JSON file < explicit overrides.
Relative paths are anchored to the repository root, independent of working directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re

from .registry import EXPERIMENTS, FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    experiment: str = "N0"
    model: str = "tiny"
    head: str = "direct"
    features: tuple[str, ...] = ("height", "tx", "band")
    bands: tuple[str, ...] = ("415", "58")
    width: int = 16
    groups: int = 8
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 2027
    loss: str = "mae"
    nlos_alpha: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    device: str = "cuda"
    amp: bool = False
    data_root: str = "/home/DataDisk/zsfang/lunarRM"
    output_dir: str = "runs/N0"
    resume: str | None = None
    target_mode: str = "raw_db"
    profile_step_m: float = 1.0
    receiver_chunk_size: int = 128
    feature_dtype: str = "float32"
    cache_dir: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["features"] = list(self.features)
        result["bands"] = list(self.bands)
        return result


def _path(value, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _integer(value, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def resolve_config(config_path=None, *, experiment=None, overrides=None) -> Config:
    values = {}
    if config_path is not None:
        with _path(config_path, "config_path").open() as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("config JSON must be an object")
    overrides = dict(overrides or {})
    requested = [x for x in (experiment, values.get("experiment"), overrides.get("experiment"))
                 if x is not None]
    if any(x != requested[0] for x in requested):
        raise ValueError("conflicting experiment selections")
    name = requested[0] if requested else "N0"
    if not isinstance(name, str) or name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {name!r}")
    allowed = {f.name for f in fields(Config)}
    unknown = (values.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    profile = EXPERIMENTS[name]
    resolved = {**asdict(Config()), **asdict(profile), "experiment": name,
                "output_dir": f"runs/{name}", **values, **overrides}
    for key in ("features", "bands"):
        sequence = resolved[key]
        if not isinstance(sequence, (list, tuple)) or any(not isinstance(x, str) for x in sequence):
            raise ValueError(f"{key} must be a list of strings")
        if len(set(sequence)) != len(sequence):
            raise ValueError(f"{key} contains duplicates")
        resolved[key] = tuple(sequence)
    for key in ("data_root", "output_dir", "resume", "cache_dir"):
        if resolved[key] is not None:
            resolved[key] = str(_path(resolved[key], key))
        elif key in ("data_root", "output_dir"):
            raise ValueError(f"{key} is required")
    cfg = Config(**resolved)
    for key in ("width", "groups", "epochs", "batch_size", "receiver_chunk_size"):
        _integer(getattr(cfg, key), key, 1)
    for key in ("num_workers", "seed"):
        _integer(getattr(cfg, key), key, 0)
    for key in ("lr", "profile_step_m"):
        _number(getattr(cfg, key), key, positive=True)
    for key in ("weight_decay", "nlos_alpha"):
        _number(getattr(cfg, key), key)
    if cfg.width % cfg.groups:
        raise ValueError("width must be divisible by GroupNorm groups")
    if cfg.model != profile.model or cfg.head != profile.head:
        raise ValueError("model/head must match the registered experiment")
    if cfg.model == "unet" and any((cfg.width * k) % min(8, cfg.width * k)
                                  for k in (1, 2, 4, 8, 16)):
        raise ValueError("width is incompatible with the N1 GroupNorm architecture")
    if set(cfg.features) - FEATURES:
        raise ValueError("unknown or forbidden input features (GT/mask must never be inputs)")
    if cfg.model == "fspl" and cfg.features:
        raise ValueError("FSPL baseline has no network input features")
    if cfg.model != "fspl" and not {"height", "tx", "band"}.issubset(cfg.features):
        raise ValueError("network features must include height, tx and band")
    if cfg.head == "residual" and "fspl" not in cfg.features:
        raise ValueError("residual experiment requires the fspl input")
    if not cfg.bands or set(cfg.bands) - {"415", "58"}:
        raise ValueError("bands must contain 415 and/or 58, without duplicates")
    if cfg.loss not in ("mae", "mse"):
        raise ValueError("loss must be mae or mse")
    if cfg.optimizer != "adamw" or cfg.scheduler != "cosine":
        raise ValueError("supported optimizer/scheduler are adamw/cosine")
    if cfg.target_mode not in ("raw_db", "official_clipped"):
        raise ValueError("unsupported target_mode")
    if cfg.feature_dtype not in ("float32", "float64"):
        raise ValueError("feature_dtype must be float32 or float64")
    if not isinstance(cfg.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", cfg.device):
        raise ValueError("device must be cpu, cuda or cuda:N")
    if type(cfg.amp) is not bool or (cfg.amp and cfg.device == "cpu"):
        raise ValueError("amp must be boolean and requires CUDA")
    data = Path(cfg.data_root)
    for key in ("output_dir", "cache_dir"):
        if getattr(cfg, key) is not None:
            dest = Path(getattr(cfg, key))
            if dest.is_relative_to(data) or data.is_relative_to(dest):
                raise ValueError(f"{key} must be separate from the read-only data root")
    return cfg


def validate_runtime(cfg: Config, *, for_training: bool = False) -> None:
    """Validate runtime requests without creating directories or loading weights.

    Schema-only configuration checks do not need CUDA. Training callers must call
    this with for_training=True before any write; checkpoint content compatibility
    is checked separately by the checkpoint loader.
    """
    if cfg.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; select cpu explicitly")
        index = int(cfg.device.partition(":")[2] or "0")
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
    if cfg.resume is not None and not Path(cfg.resume).is_file():
        raise ValueError(f"resume checkpoint not found: {cfg.resume}")
    dest = Path(cfg.output_dir)
    if dest.exists() and not dest.is_dir():
        raise ValueError("output_dir exists and is not a directory")
    if for_training and dest.exists() and cfg.resume is None:
        raise ValueError("output_dir already exists; select a new directory or resume explicitly")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check experiment configuration without training")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    for name in ("device", "data-root", "output-dir", "resume", "loss"):
        parser.add_argument(f"--{name}")
    for name in ("epochs", "batch-size", "width", "groups", "seed", "num-workers"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--bands", nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check schema only (default)")
    mode.add_argument("--dry-run", action="store_true", help="also validate device and planned run paths")
    args = vars(parser.parse_args(argv))
    path, name = args.pop("config"), args.pop("experiment")
    dry_run = args.pop("dry_run")
    args.pop("check")
    try:
        cfg = resolve_config(path, experiment=name, overrides={k: v for k, v in args.items() if v is not None})
        if dry_run:
            validate_runtime(cfg, for_training=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"mode": "dry-run" if dry_run else "check", "training_started": False,
                      "config": cfg.to_dict()}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
