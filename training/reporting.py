"""Fixed-population diagnostics and prediction-independent example panels."""

import math
import numpy as np
import torch

from training.metrics import BandMetrics
from data.features import fspl_db, BAND_FREQUENCIES_HZ


class GroupedMetrics:
    def __init__(self, coverage_edges=(0.5, 0.9)):
        self.edges = tuple(coverage_edges)
        if any(not math.isfinite(v) or not 0 < v < 1 for v in self.edges) or tuple(sorted(set(self.edges))) != self.edges:
            raise ValueError("coverage edges must be strictly increasing inside (0,1)")
        self.groups = {k: BandMetrics() for k in
                       ["los", "nlos", "single", "multi", *[f"coverage_{i}" for i in range(len(self.edges)+1)]]}
        self.terrains = {}

    def update(self, pred, target, mask, bands, los, groups, terrains):
        if los.shape != mask.shape or los.dtype != torch.bool:
            raise ValueError("geometric LoS must be an independent boolean map matching valid_mask")
        if len(groups) != len(bands) or len(terrains) != len(bands) or set(groups) - {"single", "multi"}:
            raise ValueError("invalid group/terrain identities")
        for i, band in enumerate(bands):
            p, y, valid = pred[i:i+1], target[i:i+1], mask[i:i+1]
            for region, selected in (("los", valid & los[i:i+1]), ("nlos", valid & ~los[i:i+1])):
                self.groups[region].update(p, y, selected, [band])
            self.groups[groups[i]].update(p, y, valid, [band])
            bucket = int(np.searchsorted(self.edges, valid.double().mean().item(), side="right"))
            self.groups[f"coverage_{bucket}"].update(p, y, valid, [band])
            self.terrains.setdefault(str(terrains[i]), BandMetrics()).update(p, y, valid, [band])

    def report(self):
        terrain_reports = {k: v.report()["bands"] for k, v in self.terrains.items()}
        equal = {}
        for band in ("415", "58"):
            rows = [r[band] for r in terrain_reports.values() if r[band]["items"]]
            valid = [r for r in rows if r["valid_pixels"]]
            equal[band] = {"terrains": len(rows), "valid_terrains": len(valid),
                           "mean_terrain_rmse_db": sum(r["rmse_db"] for r in valid)/len(valid) if valid else None,
                           "mean_terrain_mae_db": sum(r["mae_db"] for r in valid)/len(valid) if valid else None}
        return {"groups": {k: v.report() for k, v in self.groups.items()}, "terrain_equal": equal,
                "definitions": {"coverage_edges": list(self.edges), "coverage_intervals": "left-closed; final interval includes 1",
                                "coverage_source": "valid_mask.mean over full map; fixed project thresholds, not official classes",
                                "los_source": "DEM geometry, distinct from validity mask",
                                "region_items": "all evaluated maps; empty_items counts maps with zero valid region pixels",
                                "terrain_equal": "mean of each terrain's pixel-aggregated RMSE/MAE; diagnostic only"}}


def fixed_indices(dataset, identities):
    if not isinstance(identities, list) or not identities:
        raise ValueError("examples must be a nonempty identity list")
    mapping = {(b, row["sample_id"]): bi*len(dataset.rows)+i
               for bi, b in enumerate(dataset.bands) for i, row in enumerate(dataset.rows)}
    result, seen = [], set()
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {"band", "sample_id"}:
            raise ValueError("each example requires band and sample_id only")
        key = (identity["band"], identity["sample_id"])
        if not all(isinstance(x, str) for x in key):
            raise ValueError("example identities must be strings")
        if key in seen:
            raise ValueError("duplicate example identity")
        if key not in mapping:
            raise ValueError(f"unknown example identity: {key}")
        seen.add(key)
        result.append(mapping[key])
    return result


@torch.inference_mode()
def render_examples(model, dataset, identities, directory, device="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indices = fixed_indices(dataset, identities)
    model.to(device).eval()
    for ordinal, index in enumerate(indices):
        item = dataset[index]
        baseline = item.get("baseline_db")
        pred = model(item["features"][None].to(device),
                     baseline[None].to(device) if baseline is not None else None)[0, 0].cpu().numpy()
        if not np.isfinite(pred).all():
            raise ValueError("non-finite example prediction")
        dem, gt = item["raw_dem_m"][0].numpy(), item["target_db"][0].numpy()
        mask = item["valid_mask"][0].numpy()
        physics = fspl_db(dem, (item["tx_row"], item["tx_col"]), BAND_FREQUENCIES_HZ[item["band"]])
        error = np.where(mask, np.abs(pred-gt), np.nan)
        stem = directory / f"example-{ordinal:03d}"
        # Fixed physical display ranges make panels comparable across models.
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        panels = [(dem, "DEM / Tx (m)", 0, 496), (gt, "GT (dB)", 20, 228),
                  (pred, "Prediction (dB)", 20, 228), (mask, "Valid mask", 0, 1),
                  (error, "Valid absolute error (dB)", 0, 50), (physics, "FSPL (dB)", 20, 228)]
        try:
            for ax, (values, title, low, high) in zip(axes.flat, panels):
                im = ax.imshow(values, vmin=low, vmax=high, origin="upper")
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
            axes[0, 0].scatter([item["tx_col"]], [item["tx_row"]], marker="x", color="red")
            fig.suptitle(f"{item['sample_id']} / {item['band']}")
            fig.savefig(stem.with_suffix(".png"), dpi=120)
        finally:
            plt.close(fig)
        np.savez(stem.with_suffix(".npz"), dem_m=dem, target_db=gt, prediction_db=pred,
                 valid_mask=mask, absolute_error_db=error, fspl_db=physics,
                 tx_row=item["tx_row"], tx_col=item["tx_col"], **identities[ordinal])
