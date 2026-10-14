"""Raw dB, float64 pixel accumulators; empty populations remain undefined."""

import math
import torch
from training.losses import validate_batch


class BandMetrics:
    def __init__(self):
        self.totals = {b: {"sse": 0.0, "sae": 0.0, "valid_pixels": 0,
                           "items": 0, "empty_items": 0} for b in ("415", "58")}

    def update(self, pred, target, mask, bands):
        validate_batch(pred, target, mask, bands)
        # Per-item reductions and fixed traversal order make accumulation
        # independent of batch boundaries; subtraction itself is float64.
        for i, band in enumerate(bands):
            selected = mask[i]
            delta = pred[i][selected].detach().cpu().double() - target[i][selected].detach().cpu().double()
            if not torch.isfinite(delta).all():
                raise FloatingPointError("non-finite valid metric prediction/target")
            row = self.totals[band]
            row["items"] += 1
            row["empty_items"] += int(delta.numel() == 0)
            row["valid_pixels"] += delta.numel()
            row["sse"] += delta.square().sum().item()
            row["sae"] += delta.abs().sum().item()
            if not math.isfinite(row["sse"]) or not math.isfinite(row["sae"]):
                raise FloatingPointError("non-finite metric accumulation")

    def report(self):
        bands = {b: {**r, "rmse_db": math.sqrt(r["sse"] / r["valid_pixels"]) if r["valid_pixels"] else None,
                     "mae_db": r["sae"] / r["valid_pixels"] if r["valid_pixels"] else None}
                 for b, r in self.totals.items()}
        scores = [r["rmse_db"] for r in bands.values()]
        return {"bands": bands, "equal_band_rmse_db": sum(scores) / 2 if all(s is not None for s in scores) else None}


def selection_score(report):
    if report.get("scope") != "full" or report.get("items_seen") != report.get("items_expected"):
        raise ValueError("best selection requires full validation coverage")
    score = report.get("equal_band_rmse_db")
    if score is None or not math.isfinite(score):
        raise ValueError("best selection requires valid pixels from both bands")
    return score
