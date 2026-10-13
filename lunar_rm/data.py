"""Project adapter exposing unmodified raw dB supervision and sample identity."""

import json
import hashlib
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import FEATURES
from .features import BAND_FREQUENCIES_HZ, FEATURE_VERSION, fspl_db, geometry_features, profile_features
from .cache import FeatureCache, cache_key
from .transforms import augment_code, normalize_height, normalize_pathloss, spatial_transform
from .storage import (BANDS, GRID, file_identity, open_array, read_index,
                      split_paths, validate_index_rows)


class LunarDataset(Dataset):
    """Read consolidated train/val arrays lazily in each worker.

    Base channels are normalized height, TX one-hot, frequency ID and optional FSPL.
    Raw arrays are always independent copies of read-only memory maps. Physical
    feature channels are added by the feature pipeline, never from supervision.
    """

    def __init__(self, root, split="train", bands=BANDS, features=("height", "tx", "band"),
                 *, augment=False, seed=2027, target_mode="raw_db", need_los=False,
                 profile_step_m=1.0, receiver_chunk_size=128, feature_dtype="float32",
                 memory_budget_mb=64, cache_dir=None):
        self.root = Path(root).resolve()
        self.split = split
        self.paths = split_paths(self.root, split)
        if split == "test":
            raise ValueError("test targets/masks are withheld; supervised test evaluation is unavailable")
        if augment and split != "train":
            raise ValueError("spatial augmentation is only allowed for train")
        if target_mode not in ("raw_db", "official_clipped"):
            raise ValueError("unsupported target_mode")
        augment_code(seed, 0, "validate_seed", "415")
        self.augment, self.seed, self.target_mode = augment, seed, target_mode
        self.need_los = need_los
        self.profile_step_m, self.receiver_chunk_size = profile_step_m, receiver_chunk_size
        self.feature_dtype, self.memory_budget_mb = str(np.dtype(feature_dtype)), memory_budget_mb
        if self.feature_dtype not in ("float32", "float64"):
            raise ValueError("feature_dtype must be float32 or float64")
        self.cache = FeatureCache(cache_dir, data_root=self.root) if cache_dir is not None else None
        # Shared spawn-context scalar keeps persistent workers at the main epoch.
        self._epoch = multiprocessing.get_context("spawn").Value("q", 0)
        self.bands = tuple(bands)
        self.feature_names = tuple(features)
        if not self.bands or len(set(self.bands)) != len(self.bands) or set(self.bands) - set(BANDS):
            raise ValueError("bands must be a nonempty unique sequence of 415/58")
        if len(set(self.feature_names)) != len(self.feature_names) or set(self.feature_names) - FEATURES:
            raise ValueError("unknown, duplicate or forbidden input feature")
        self.rows = read_index(self.paths["index"])
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("grid") != list(GRID):
            raise ValueError("expected official 256x256 grid")
        if self.metadata.get("heightmap_range_m") != [0, 496] or self.metadata.get("pathloss_range_db") != [20, 228]:
            raise ValueError("expected official common normalization ranges")
        if self.metadata.get("bands_hz") != BAND_FREQUENCIES_HZ or self.metadata.get("resolution_m_per_px") != 1:
            raise ValueError("expected official frequencies and metre grid spacing")
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

    def set_epoch(self, epoch):
        """Call between complete DataLoader iterations, before requesting batches."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch.value = epoch

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

    def _physical(self, dem, tx, row, band, code):
        geometry_names = {"relative_row", "relative_col", "relative_height", "slope", "curvature_proxy"}
        want_geometry = bool(set(self.feature_names) & geometry_names)
        want_profiles = self.need_los or bool(set(self.feature_names) & {"los", "fresnel"})
        want_baseline = "fspl" in self.feature_names or not self.feature_names
        expected = ({"fspl"} if want_baseline else set()) | (geometry_names if want_geometry else set())
        if want_profiles:
            expected |= {"los", "fresnel", "clearance_m"}
        if not expected:
            return {}
        identity = {"version": FEATURE_VERSION, "dtype": self.feature_dtype,
                    "features": sorted(expected), "step_m": self.profile_step_m,
                    "chunk_size": self.receiver_chunk_size, "memory_budget_mb": self.memory_budget_mb,
                    "tx": list(tx), "terrain_id": row["terrain_id"], "sample_id": row["sample_id"],
                    "frequency_hz": BAND_FREQUENCIES_HZ[band] if want_baseline or want_profiles else None,
                    "band": band if want_baseline or want_profiles else None,
                    "augmentation_code": code, "spacing_m": self.metadata["resolution_m_per_px"],
                    "dem_sha256": hashlib.sha256(dem.tobytes()).hexdigest(),
                    "input_files": {k: v for k, v in self.identity["files"].items()
                                    if k.endswith(("hm.npy", "index.csv")) or k == "metadata.json"}}
        key = cache_key(identity)
        if self.cache is not None:
            cached = self.cache.get(key, identity, expected, dem.shape)
            if cached is not None:
                return cached
        common = {"spacing_m": identity["spacing_m"], "dtype": self.feature_dtype}
        result = {}
        if want_baseline:
            result["fspl"] = fspl_db(dem, tx, BAND_FREQUENCIES_HZ[band], **common)
        if want_geometry:
            result.update(geometry_features(dem, tx, **common))
        if want_profiles:
            result.update(profile_features(dem, tx, BAND_FREQUENCIES_HZ[band], **common,
                                           step_m=self.profile_step_m, chunk_size=self.receiver_chunk_size,
                                           memory_budget_mb=self.memory_budget_mb))
        if self.cache is not None:
            self.cache.put(key, identity, result)
        return result

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
        original_r, original_c = r, c
        code = augment_code(self.seed, self._epoch.value, row["sample_id"], band) if self.augment else 0
        if code:
            dem, target, mask, tx = (spatial_transform(a, code) for a in (dem, target, mask, tx))
            r, c = (int(v) for v in np.argwhere(tx == 1)[0])
        channel = {"height": normalize_height(dem), "tx": tx,
                   "band": np.full(GRID, float(band == "58"), dtype=np.float32)}
        physical = self._physical(dem, (r, c), row, band, code)
        physics = {}
        for name, value in physical.items():
            if name == "fspl":
                channel[name] = normalize_pathloss(value)
                physics["baseline_db"] = torch.from_numpy(value.astype(np.float32)[None])
            elif name in ("relative_row", "relative_col"):
                channel[name] = value / 256.0  # metre offsets / fixed map width in metres
            elif name == "relative_height":
                channel[name] = value / 496.0
            elif name != "clearance_m":
                channel[name] = value.astype(np.float32)
            if name == "los":
                physics["geometric_los"] = torch.from_numpy(value[None])
        features = np.stack([channel[k] for k in self.feature_names]).astype(np.float32) if self.feature_names else np.empty((0, *GRID), dtype=np.float32)
        training_target = np.clip(target, 20, 228) if self.target_mode == "official_clipped" else target.copy()
        return {**physics, "raw_dem_m": torch.from_numpy(dem[None]), "tx_map": torch.from_numpy(tx[None]),
                "target_db": torch.from_numpy(target[None]), "valid_mask": torch.from_numpy(mask[None]),
                "training_target_db": torch.from_numpy(training_target[None]),
                "clipped_valid_pixels": int(np.count_nonzero(mask & ((target < 20) | (target > 228)))) if self.target_mode == "official_clipped" else 0,
                "augmentation_code": code,
                "features": torch.from_numpy(features), "sample_id": row["sample_id"],
                "terrain_id": row["terrain_id"], "band": band, "split": self.split,
                "index_row": row_idx, "hm_row": hm_row, "tx_index": int(row["tx_index"]),
                "group": row["group"], "tx_row": r, "tx_col": c,
                "original_tx_row": original_r, "original_tx_col": original_c}


def dataset_from_config(cfg, split, *, augment=False, need_los=False):
    return LunarDataset(cfg.data_root, split=split, bands=cfg.bands, features=cfg.features,
                        augment=augment, seed=cfg.seed, target_mode=cfg.target_mode,
                        need_los=need_los or cfg.nlos_alpha > 0,
                        profile_step_m=cfg.profile_step_m, receiver_chunk_size=cfg.receiver_chunk_size,
                        feature_dtype=cfg.feature_dtype, cache_dir=cfg.cache_dir)
