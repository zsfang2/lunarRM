"""Tiny depthwise-separable U-Net and physically explicit dB output heads."""

import torch
from torch import nn
from torch.nn import functional as F

from model import LunarUNet


class SeparableBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, groups):
        layers = []
        for source in (in_channels, out_channels):
            layers.extend([nn.Conv2d(source, source, 3, padding=1, groups=source, bias=False),
                           nn.Conv2d(source, out_channels, 1, bias=False),
                           nn.GroupNorm(groups, out_channels), nn.SiLU()])
        super().__init__(*layers)


class TinyUNet(nn.Module):
    def __init__(self, in_channels, width=16, groups=8):
        super().__init__()
        if in_channels <= 0 or width <= 0 or groups <= 0 or width % groups:
            raise ValueError("invalid input channels or GroupNorm width/groups")
        widths = [width * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([SeparableBlock(source, target, groups)
                                      for source, target in zip([in_channels]+widths[:-1], widths)])
        self.decoders = nn.ModuleList([SeparableBlock(widths[i+1]+widths[i], widths[i], groups)
                                      for i in (3, 2, 1, 0)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)


class RadioMapModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.signature = {"features": tuple(cfg.features), "model": cfg.model, "head": cfg.head,
                          "width": cfg.width, "groups": cfg.groups}
        self.model_kind, self.output_kind = cfg.model, cfg.head
        self.channels = len(cfg.features)
        if cfg.model == "tiny":
            self.backbone = TinyUNet(self.channels, cfg.width, cfg.groups)
        elif cfg.model == "unet":
            self.backbone = LunarUNet(in_channels=self.channels, base_channels=cfg.width)
        elif cfg.model == "fspl":
            self.backbone = None
        else:
            raise ValueError("unsupported model")

    def get_extra_state(self):
        return self.signature.copy()

    def set_extra_state(self, state):
        if state != self.signature:
            raise ValueError("checkpoint model signature/feature order mismatch")

    def forward(self, features, baseline_db=None):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} input channels in BCHW")
        expected = (features.shape[0], 1, *features.shape[-2:])
        if self.output_kind in ("residual", "baseline"):
            if baseline_db is None or tuple(baseline_db.shape) != expected:
                raise ValueError("baseline_db must have matching (B,1,H,W) shape")
            if baseline_db.device != features.device:
                raise ValueError("baseline and features must use the same device")
        if self.output_kind == "baseline":
            return baseline_db.float()
        if min(features.shape[-2:]) < 16:
            raise ValueError("U-Net spatial dimensions must be at least 16")
        value = self.backbone(features).float()
        if self.output_kind == "residual":
            return baseline_db.float() + 208.0 * value
        return 20.0 + 208.0 * value


def build_model(cfg):
    return RadioMapModel(cfg)


def parameter_report(model):
    count = sum(p.numel() for p in model.parameters())
    return {"parameters": count, "model": model.model_kind,
            "tiny_target_met": (100000 <= count <= 500000) if model.model_kind == "tiny" else None,
            "tiny_target_range": [100000, 500000]}
