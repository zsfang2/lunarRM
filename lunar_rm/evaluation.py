"""Complete validation or explicitly labelled fixed per-band subsets."""

import torch
from torch.utils.data import DataLoader, Subset
from .metrics import BandMetrics


def evaluation_indices(dataset, max_per_band=None):
    if max_per_band is not None and (type(max_per_band) is not int or max_per_band < 1):
        raise ValueError("max_per_band must be a positive integer")
    rows = len(dataset.rows)
    count = rows if max_per_band is None else min(rows, max_per_band)
    return [b * rows + i for b in range(len(dataset.bands)) for i in range(count)]


def evaluate_model(model, dataset, *, batch_size=16, device="cpu", num_workers=0, max_per_band=None):
    if dataset.split != "val" or dataset.augment:
        raise ValueError("evaluation requires unaugmented fixed val data")
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
        raise ValueError("invalid evaluation batch size/worker count")
    indices = evaluation_indices(dataset, max_per_band)
    expected = [(dataset.bands[i // len(dataset.rows)], dataset.rows[i % len(dataset.rows)]["sample_id"])
                for i in indices]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate evaluation identity")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers,
                        **({"multiprocessing_context": "spawn", "timeout": 120} if num_workers else {}))
    was_training = model.training
    model.to(device).eval()
    metrics, seen = BandMetrics(), []
    try:
        with torch.inference_mode():
            for batch in loader:
                features = batch["features"].to(device)
                baseline = batch.get("baseline_db")
                pred = model(features, baseline.to(device) if baseline is not None else None)
                metrics.update(pred.cpu(), batch["target_db"], batch["valid_mask"], batch["band"])
                seen.extend(zip(batch["band"], batch["sample_id"]))
    finally:
        model.train(was_training)
    if seen != expected:
        raise ValueError("missing, duplicate or unexpected validation identities")
    return {**metrics.report(), "scope": "full" if max_per_band is None else "subset",
            "items_seen": len(seen), "items_expected": len(expected),
            "full_dataset_items": len(dataset), "max_per_band": max_per_band}


def evaluate_checkpoint(cfg, checkpoint=None, *, batch_size=None, max_per_band=None):
    from .checkpoint import read_checkpoint, validate_inference
    from .config import validate_runtime
    from .data import dataset_from_config
    from .models import build_model
    from .provenance import code_identity
    from .storage import file_identity
    validate_runtime(cfg)
    if cfg.model != "fspl" and checkpoint is None:
        raise ValueError("learned models require a checkpoint for evaluation")
    dataset = dataset_from_config(cfg, "val")
    model = build_model(cfg)
    if checkpoint is not None:
        state = read_checkpoint(checkpoint)
        validate_inference(state, cfg, dataset)
        model.load_state_dict(state["model"], strict=True)
    metrics = evaluate_model(model, dataset, batch_size=cfg.batch_size if batch_size is None else batch_size,
                             device=cfg.device, num_workers=cfg.num_workers, max_per_band=max_per_band)
    return {"metrics": metrics, "config": cfg.to_dict(), "data_identity": {"val": dataset.identity},
            "code_identity": code_identity(), "checkpoint": file_identity(checkpoint, content_hash=True) if checkpoint else None,
            "target": "raw_db", "training_started": False}
