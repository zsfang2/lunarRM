"""Deterministic balanced batches for contiguous, equally sized band blocks."""

import math
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, rows_per_band, num_bands, batch_size, seed=2027):
        for name, value in (("rows_per_band", rows_per_band), ("num_bands", num_bands),
                            ("batch_size", batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if batch_size % num_bands:
            raise ValueError("batch_size must be divisible by num_bands")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.rows_per_band, self.num_bands = rows_per_band, num_bands
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.rows_per_band / (self.batch_size // self.num_bands))

    def __iter__(self):
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % (2**63))
        orders = [torch.randperm(self.rows_per_band, generator=generator).tolist()
                  for _ in range(self.num_bands)]
        per_band = self.batch_size // self.num_bands
        for start in range(0, self.rows_per_band, per_band):
            yield [orders[band][row] + band * self.rows_per_band
                   for row in range(start, min(start + per_band, self.rows_per_band))
                   for band in range(self.num_bands)]

    def state_dict(self):
        return {"rows_per_band": self.rows_per_band, "num_bands": self.num_bands,
                "batch_size": self.batch_size, "seed": self.seed, "epoch": self.epoch,
                "tail_policy": "keep_balanced", "resume_granularity": "epoch_boundary"}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected) or any(state[k] != v for k, v in expected.items() if k != "epoch"):
            raise ValueError("sampler configuration mismatch")
        self.set_epoch(state["epoch"])
