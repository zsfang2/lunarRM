"""Train and evaluate a masked-RMSE U-Net baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import LunarUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/home/DataDisk/zsfang/lunarRM"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0 uses the full epoch; positive values are useful for CPU trials")
    p.add_argument("--max-val-batches", type=int, default=0,
                   help="0 evaluates the full validation set")
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_official_dataset(data_root: Path):
    path = data_root / "lunar_dataset.py"
    spec = importlib.util.spec_from_file_location("official_lunar_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official dataset module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LunarRadioMapDataset


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: str):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def validate(model, loader, device, db_scale, max_batches=0):
    model.eval()
    sums = {"415": 0.0, "58": 0.0}
    counts = {"415": 0.0, "58": 0.0}
    batches = {"415": 0, "58": 0}
    for x, y, mask, names in loader:
        present = {band for band in ("415", "58")
                   if any(name.endswith(f"_{band}") for name in names)}
        eligible = {band for band in present
                    if not max_batches or batches[band] < max_batches}
        if not eligible:
            if max_batches and all(value >= max_batches for value in batches.values()):
                break
            continue
        for band in eligible:
            batches[band] += 1
        if max_batches and all(value >= max_batches for value in batches.values()):
            final_batch = True
        else:
            final_batch = False
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(x)
        squared_db = ((pred - y) * db_scale).square() * mask
        for band in ("415", "58"):
            if band not in eligible:
                continue
            select = torch.tensor([name.endswith(f"_{band}") for name in names], device=device)
            if select.any():
                sums[band] += squared_db[select].sum().item()
                counts[band] += mask[select].sum().item()
        if final_batch:
            break
    rmse = {
        band: math.sqrt(sums[band] / counts[band]) if counts[band] else float("nan")
        for band in sums
    }
    rmse["mean"] = 0.5 * (rmse["415"] + rmse["58"])
    return rmse


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step,
                    best_rmse, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_rmse_db": best_rmse,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Dataset = load_official_dataset(args.data_root)
    train_set = Dataset(args.data_root, split="train", band="both", normalize=True,
                        augment=True, return_mask=True, return_name=True)
    val_set = Dataset(args.data_root, split="val", band="both", normalize=True,
                      augment=False, return_mask=True, return_name=True)
    db_scale = float(train_set.pl_max - train_set.pl_min)
    train_loader = make_loader(train_set, args.batch_size, args.num_workers, True, args.device)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers, False, args.device)

    model = LunarUNet(base_channels=args.base_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    start_epoch, global_step, best_rmse = 1, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_rmse = checkpoint["best_rmse_db"]

    parameter_count = sum(p.numel() for p in model.parameters())
    config = vars(args).copy()
    config.update(parameters=parameter_count, train_items=len(train_set), val_items=len(val_set))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        epoch_start = time.time()
        steps = 0
        for batch_idx, (x, y, mask, _) in enumerate(train_loader):
            if args.max_train_steps and batch_idx >= args.max_train_steps:
                break
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                pred = model(x)
                loss = masked_mse(pred, y, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            steps += 1
            global_step += 1
            if steps == 1 or steps % 25 == 0:
                print(f"epoch={epoch} step={steps} loss={loss_sum / steps:.6f}", flush=True)
        scheduler.step()
        rmse = validate(model, val_loader, args.device, db_scale, args.max_val_batches)
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_masked_mse_normalized": loss_sum / max(steps, 1),
            "val_rmse_415_db": rmse["415"],
            "val_rmse_58_db": rmse["58"],
            "val_rmse_mean_db": rmse["mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler,
                        scaler, epoch, global_step,
                        min(best_rmse, rmse["mean"]), args)
        if rmse["mean"] < best_rmse:
            best_rmse = rmse["mean"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, scheduler,
                            scaler, epoch, global_step, best_rmse, args)


if __name__ == "__main__":
    main()
