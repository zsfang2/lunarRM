"""Read one official sample batch and run inference without training."""

from pathlib import Path

import torch

from model import LunarUNet
from train import load_official_dataset, make_loader, masked_mse


DATA_ROOT = Path("/home/DataDisk/zsfang/lunarRM")


def main():
    Dataset = load_official_dataset(DATA_ROOT)
    dataset = Dataset(
        DATA_ROOT,
        split="val",
        band="both",
        normalize=True,
        augment=False,
        return_mask=True,
        return_name=True,
    )
    loader = make_loader(dataset, batch_size=2, workers=0, shuffle=False, device="cpu")
    x, y, mask, names = next(iter(loader))
    model = LunarUNet(base_channels=16).eval()
    with torch.inference_mode():
        prediction = model(x)
        loss = masked_mse(prediction, y, mask)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"dataset_layout={dataset.layout}")
    print(f"dataset_items={len(dataset)}")
    print(f"names={list(names)}")
    print(f"input={tuple(x.shape)} target={tuple(y.shape)} mask={tuple(mask.shape)}")
    print(f"prediction={tuple(prediction.shape)} parameters={parameters:,}")
    print(f"initial_masked_mse={loss.item():.6f}")


if __name__ == "__main__":
    main()

