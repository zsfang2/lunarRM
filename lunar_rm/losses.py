"""Masked dB objectives: average pixels within a band, then average bands."""

import math
import torch


def validate_batch(pred, target, mask, bands):
    if pred.shape != target.shape or pred.shape != mask.shape or pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError("pred, target and mask must have matching B1HW shapes")
    if mask.dtype != torch.bool or pred.device != target.device or pred.device != mask.device:
        raise ValueError("mask must be boolean and all tensors must share a device")
    if len(bands) != len(pred) or any(b not in ("415", "58") for b in bands):
        raise ValueError("one supported band string is required per item")


def masked_band_loss(pred, target, mask, bands, *, kind="mae", geometric_los=None, nlos_alpha=0.0):
    validate_batch(pred, target, mask, bands)
    if kind not in ("mae", "mse") or not math.isfinite(nlos_alpha) or nlos_alpha < 0:
        raise ValueError("invalid loss kind or NLoS weight")
    if nlos_alpha > 0 and geometric_los is None:
        raise ValueError("geometric LoS is required for NLoS weighting")
    if geometric_los is not None and (geometric_los.shape != mask.shape or
                                     geometric_los.dtype != torch.bool or geometric_los.device != mask.device):
        raise ValueError("geometric LoS must match the boolean mask shape/device")
    counts = mask.flatten(1).sum(1)
    result = {"skipped_items": int((counts == 0).sum()), "valid_pixels": int(counts.sum()),
              "error": None, "loss": None}
    terms, nlos_terms = [], []
    for band in ("415", "58"):
        rows = torch.tensor([b == band for b in bands], device=mask.device, dtype=torch.bool)
        selected = mask & rows[:, None, None, None]
        if not selected.any():
            continue
        # Select before arithmetic: invalid NaN/Inf must not poison gradients.
        error = pred[selected].float() - target[selected].float()
        if not torch.isfinite(error).all():
            raise FloatingPointError("non-finite valid prediction/target")
        values = error.abs() if kind == "mae" else error.square()
        terms.append(values.mean())
        if nlos_alpha > 0:
            nlos = ~geometric_los[selected]
            if nlos.any():
                nlos_terms.append(values[nlos].mean())
    if terms:
        error = torch.stack(terms).mean()
        if nlos_terms:
            error = error + nlos_alpha * torch.stack(nlos_terms).mean()
        if not torch.isfinite(error):
            raise FloatingPointError("non-finite loss reduction")
        result.update(error=error, loss=error / (208 if kind == "mae" else 208**2))
    return result
