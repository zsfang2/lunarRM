"""Explicit training, complete validation and epoch-boundary checkpoints."""

import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import atomic_save, read_checkpoint, restore_rng, rng_state, validate_resume
from .config import resolve_config, validate_runtime
from .data import dataset_from_config
from .evaluation import evaluate_model
from .losses import masked_band_loss
from .metrics import selection_score
from .models import build_model
from .provenance import code_identity
from .sampling import BalancedBatchSampler


def train(cfg, *, until_epoch=None):
    """Run training only when called explicitly; until_epoch stops at a saved boundary.

    The configured total epochs must remain fixed when resuming cosine scheduling.
    Exact replay is tested on CPU with one thread; GPU replay is not guaranteed.
    """
    cfg = resolve_config(overrides=cfg.to_dict())
    validate_runtime(cfg, for_training=True)
    if cfg.model == "fspl":
        raise ValueError("A0 has no trainable parameters; use evaluation")
    if until_epoch is not None and (type(until_epoch) is not int or not 1 <= until_epoch <= cfg.epochs):
        raise ValueError("until_epoch must be within the configured budget")
    if cfg.num_workers and os.environ.get("MKL_THREADING_LAYER") == "INTEL":
        raise ValueError("spawn workers require a compatible MKL backend; launch with MKL_THREADING_LAYER=GNU")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % 2**32)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    train_set = dataset_from_config(cfg, "train", augment=True)
    val_set = dataset_from_config(cfg, "val")
    sampler = BalancedBatchSampler(len(train_set.rows), len(cfg.bands), cfg.batch_size, cfg.seed)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg.num_workers,
                        generator=torch.Generator().manual_seed(cfg.seed),
                        **({"multiprocessing_context": "spawn", "timeout": 120} if cfg.num_workers else {}))
    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = (torch.amp.GradScaler("cuda", enabled=cfg.amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=cfg.amp))
    start, global_step, best_metric, best_report = 1, 0, None, None
    output = Path(cfg.output_dir)
    if cfg.resume:
        state = read_checkpoint(cfg.resume)
        validate_resume(state, cfg, train_set, val_set)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
        start, global_step = state["epoch"] + 1, state["global_step"]
        best_metric, best_report = state["best_metric"], state["best_report"]
        # Existing output can only continue its latest committed epoch.
        if output.exists():
            if Path(cfg.resume).resolve() != (output / "last.pt").resolve():
                raise ValueError("existing output must resume its own last.pt; branch to a new directory otherwise")
            logs = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            if not logs or logs[-1]["epoch"] != state["epoch"] or logs[-1]["global_step"] != global_step:
                raise ValueError("log/checkpoint boundary mismatch; resume into a new output directory")
    end = cfg.epochs if until_epoch is None else until_epoch
    if start > end:
        raise ValueError("no epochs remain in the requested interval")
    identity = code_identity()
    data_identity = {"train": train_set.identity, "val": val_set.identity}
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        with (output / "config.json").open("x") as stream:
            json.dump({"config": cfg.to_dict(), "data_identity": data_identity, "code_identity": identity,
                       "resume_granularity": "epoch_boundary", "torch_version": str(torch.__version__),
                       "threads": torch.get_num_threads()}, stream, indent=2, allow_nan=False)
    last = None
    for epoch in range(start, end + 1):
        model.train()
        sampler.set_epoch(epoch - 1)
        train_set.set_epoch(epoch - 1)
        loader.generator.manual_seed(cfg.seed + epoch)
        updates, skipped, valid, empty_batches, clipped = 0, 0, 0, 0, 0
        loss_sum, error_sum = 0.0, 0.0
        lr = optimizer.param_groups[0]["lr"]
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            tensors = {k: v.to(cfg.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type=torch.device(cfg.device).type, enabled=cfg.amp):
                pred = model(tensors["features"], tensors.get("baseline_db"))
            result = masked_band_loss(pred, tensors["training_target_db"], tensors["valid_mask"], batch["band"],
                                      kind=cfg.loss, geometric_los=tensors.get("geometric_los"), nlos_alpha=cfg.nlos_alpha)
            skipped += result["skipped_items"]
            valid += result["valid_pixels"]
            clipped += int(batch["clipped_valid_pixels"].sum())
            if result["loss"] is None:
                empty_batches += 1
                continue
            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            global_step += 1
            loss_sum += result["loss"].item()
            error_sum += result["error"].item()
        # The schedule advances per epoch even when every training mask is empty.
        # Explicit epoch avoids the optimizer-order warning on a zero-update epoch.
        if updates:
            scheduler.step()
        else:
            scheduler.last_epoch += 1
            for group, value in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
                group["lr"] = value
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            scheduler._step_count += 1
        validation = evaluate_model(model, val_set, batch_size=cfg.batch_size, device=cfg.device, num_workers=cfg.num_workers)
        score = selection_score(validation) if validation["equal_band_rmse_db"] is not None else None
        improved = score is not None and (best_metric is None or score < best_metric)
        if improved:
            best_metric, best_report = score, validation
        last = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "global_step": global_step, "best_metric": best_metric, "best_report": best_report,
                "rng": rng_state(), "sampler": sampler.state_dict(), "config": cfg.to_dict(),
                "data_identity": data_identity, "code_identity": identity, "validation": validation,
                "resume_granularity": "epoch_boundary",
                "runtime": {"torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
                            "device": cfg.device, "amp": cfg.amp}}
        if improved:
            atomic_save(last, output / "best.pt")
        atomic_save(last, output / "last.pt")
        log = {"epoch": epoch, "lr": lr, "next_lr": optimizer.param_groups[0]["lr"], "updates": updates,
               "global_step": global_step, "valid_pixels": valid, "skipped_items": skipped,
               "empty_batches": empty_batches, "clipped_valid_pixels": clipped,
               "mean_batch_loss": loss_sum / updates if updates else None,
               "mean_batch_error": error_sum / updates if updates else None,
               "error_units": "dB" if cfg.loss == "mae" else "dB^2", "validation": validation}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(log, allow_nan=False), flush=True)
    return last
