"""Versioned checkpoint reading and compatibility checks."""

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
    return state


def validate_inference(state, cfg, dataset):
    expected = cfg.to_dict()
    mismatches = [key for key in INFERENCE_KEYS if state["config"].get(key) != expected[key]]
    if mismatches:
        raise ValueError(f"checkpoint config mismatch: {mismatches}")
    if state["data_identity"].get(dataset.split) != dataset.identity:
        raise ValueError("checkpoint data identity mismatch")
