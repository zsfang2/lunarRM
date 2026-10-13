"""Project adapter exposing unmodified raw dB supervision and sample identity."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import FEATURES
from .storage import (BANDS, GRID, file_identity, open_array, read_index,
                      split_paths, validate_index_rows)


class LunarDataset(Dataset):
    """Read consolidated train/val arrays lazily in each worker.

    Current base channels are normalized height, TX one-hot and frequency ID.
    Raw arrays are always independent copies of read-only memory maps. Physical
    feature channels are added by the feature pipeline, never from supervision.
    """

    def __init__(self, root, split="train", bands=BANDS, features=("height", "tx", "band")):
        self.root = Path(root).resolve()
        self.split = split
        self.paths = split_paths(self.root, split)
        if split == "test":
            raise ValueError("test targets/masks are withheld; supervised test evaluation is unavailable")
        self.bands = tuple(bands)
        self.feature_names = tuple(features)
        if not self.bands or len(set(self.bands)) != len(self.bands) or set(self.bands) - set(BANDS):
            raise ValueError("bands must be a nonempty unique sequence of 415/58")
        if len(set(self.feature_names)) != len(self.feature_names) or set(self.feature_names) - FEATURES:
            raise ValueError("unknown, duplicate or forbidden input feature")
        if set(self.feature_names) - {"height", "tx", "band"}:
            raise ValueError("physical input channels require the physical feature pipeline")
        self.rows = read_index(self.paths["index"])
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("grid") != list(GRID):
            raise ValueError("expected official 256x256 grid")
        if self.metadata.get("heightmap_range_m") != [0, 496] or self.metadata.get("pathloss_range_db") != [20, 228]:
            raise ValueError("expected official common normalization ranges")
        self._arrays = {}
        self._pid = os.getpid()
        self.identity = {"split": split, "files": {}}
        for name in ("metadata.json", "manifest.json", "lunar_dataset.py"):
            path = self.root / name
            if path.is_file():
                self.identity["files"][name] = file_identity(path, content_hash=True)
        index_path = self.paths["index"]
        self.identity["files"][str(index_path.relative_to(self.root))] = file_identity(index_path, content_hash=True)
        n = len(self.rows)
        terrain_count = len({r["terrain_id"] for r in self.rows})
        errors = []
        validate_index_rows(self.rows, terrain_count, errors, split)
        if errors:
            raise ValueError("; ".join(errors[:10]))
        for field in ("hm", *(f"{kind}{band}" for band in self.bands for kind in ("pl", "mask"))):
            path = self.paths[field]
            array = open_array(path)
            shape = (terrain_count, *GRID) if field == "hm" else ((n, 8192) if field.startswith("mask") else (n, *GRID))
            dtype = "float32" if field == "hm" else ("uint8" if field.startswith("mask") else "float16")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(f"{field}: expected {shape} {dtype}, got {array.shape} {array.dtype}")
            self.identity["files"][str(path.relative_to(self.root))] = {
                **file_identity(path), "shape": list(array.shape), "dtype": str(array.dtype)}

    def __len__(self):
        return len(self.rows) * len(self.bands)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}  # spawn must reopen maps, never pickle their contents
        state["_pid"] = None
        return state

    def _array(self, field):
        if self._pid != os.getpid():
            self._arrays = {}
            self._pid = os.getpid()
        if field not in self._arrays:
            self._arrays[field] = open_array(self.paths[field])
        return self._arrays[field]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        band_idx, row_idx = divmod(index, len(self.rows))
        band, row = self.bands[band_idx], self.rows[row_idx]
        hm_row = int(row["hm_row"])
        r, c = int(row["tx_row"]), int(row["tx_col"])
        dem = np.array(self._array("hm")[hm_row], dtype=np.float32, copy=True)
        target = np.array(self._array(f"pl{band}")[row_idx], dtype=np.float32, copy=True)
        mask = np.unpackbits(self._array(f"mask{band}")[row_idx], bitorder="big").reshape(GRID).astype(bool)
        if not np.isfinite(dem).all() or not np.isfinite(target).all():
            raise ValueError(f"non-finite DEM/target: {self.split}/{row['sample_id']}/{band}")
        tx = np.zeros(GRID, dtype=np.float32)
        tx[r, c] = 1
        channel = {"height": np.clip(dem / 496.0, 0, 1), "tx": tx,
                   "band": np.full(GRID, float(band == "58"), dtype=np.float32)}
        features = np.stack([channel[k] for k in self.feature_names]) if self.feature_names else np.empty((0, *GRID), dtype=np.float32)
        return {"raw_dem_m": torch.from_numpy(dem[None]), "tx_map": torch.from_numpy(tx[None]),
                "target_db": torch.from_numpy(target[None]), "valid_mask": torch.from_numpy(mask[None]),
                "features": torch.from_numpy(features), "sample_id": row["sample_id"],
                "terrain_id": row["terrain_id"], "band": band, "split": self.split,
                "index_row": row_idx, "hm_row": hm_row, "tx_index": int(row["tx_index"]),
                "group": row["group"], "tx_row": r, "tx_col": c,
                "original_tx_row": r, "original_tx_col": c}
