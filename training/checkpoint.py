"""Versioned checkpoint reading and compatibility checks."""

import os
import math
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


INFERENCE_KEYS = ("model", "head", "features", "bands", "width", "groups",
                  "profile_step_m", "feature_dtype")


def read_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"unreadable checkpoint: {path}") from exc
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    for key in ("model", "config", "data_identity"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"checkpoint missing {key}")
    from models.presets import LEGACY_NAMES
    name = state["config"].get("experiment")
    state["config"]["experiment"] = LEGACY_NAMES.get(name, name)
    return state


def validate_inference(state, cfg, dataset):
    expected = cfg.to_dict()
    mismatches = [key for key in INFERENCE_KEYS if state["config"].get(key) != expected[key]]
    if mismatches:
        raise ValueError(f"checkpoint config mismatch: {mismatches}")
    if state["data_identity"].get(dataset.split) != dataset.identity:
        raise ValueError("checkpoint data identity mismatch")


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            name = stream.name
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume(state, cfg, train_set, val_set):
    validate_inference(state, cfg, val_set)
    if state["data_identity"].get("train") != train_set.identity:
        raise ValueError("checkpoint train data identity mismatch")
    ignored = {"output_dir", "resume", "cache_dir", "data_root", "num_workers"}
    expected = cfg.to_dict()
    if set(state["config"]) != set(expected) or any(state["config"][k] != v for k, v in expected.items() if k not in ignored):
        raise ValueError("resume config mismatch (including original total epochs)")
    required = {"optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric",
                "best_report", "rng", "sampler", "validation", "code_identity", "resume_granularity"}
    if required - state.keys() or state.get("resume_granularity") != "epoch_boundary":
        raise ValueError("checkpoint lacks complete epoch-boundary resume state")
    epoch, steps = state["epoch"], state["global_step"]
    if type(epoch) is not int or not 1 <= epoch <= cfg.epochs or type(steps) is not int or steps < 0:
        raise ValueError("invalid checkpoint epoch/update count")
    scheduler = state["scheduler"]
    if scheduler.get("last_epoch") != epoch or scheduler.get("T_max") != cfg.epochs:
        raise ValueError("scheduler state does not match saved epoch/budget")
    expected_lr = cfg.lr * (1 + math.cos(math.pi * epoch / cfg.epochs)) / 2
    if (scheduler.get("base_lrs") != [cfg.lr] or scheduler.get("eta_min") != 0 or
            scheduler.get("_step_count") != epoch + 1 or
            len(scheduler.get("_last_lr", [])) != 1 or
            not math.isclose(scheduler["_last_lr"][0], expected_lr, rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError("scheduler was reset or its schedule changed")
    if state["sampler"].get("epoch") != epoch - 1:
        raise ValueError("sampler epoch does not match checkpoint")
    groups = state["optimizer"].get("param_groups", [])
    if len(groups) != 1 or [g.get("lr") for g in groups] != scheduler.get("_last_lr"):
        raise ValueError("optimizer/scheduler learning rate mismatch")
    if groups[0].get("weight_decay") != cfg.weight_decay or groups[0].get("betas") != (0.9, 0.999):
        raise ValueError("optimizer hyperparameters changed")
    opt_states = state["optimizer"].get("state", {})
    if steps and (set(opt_states) != set(groups[0]["params"]) or
                  any(float(s.get("step", -1)) != steps for s in opt_states.values())):
        raise ValueError("optimizer update state missing or reset")
