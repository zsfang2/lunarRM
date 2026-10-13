"""Public affine scales and deterministic, aligned spatial transforms."""

import hashlib
import json

import numpy as np


HEIGHT_MAX_M = 496.0
PL_OFFSET_DB = 20.0
PL_SCALE_DB = 208.0


def normalize_height(dem_m):
    return np.clip(np.asarray(dem_m, dtype=np.float32) / HEIGHT_MAX_M, 0, 1)


def normalize_pathloss(target_db, *, clip=False):
    value = (np.asarray(target_db, dtype=np.float32) - PL_OFFSET_DB) / PL_SCALE_DB
    return np.clip(value, 0, 1) if clip else value


def denormalize_pathloss(value):
    return np.asarray(value, dtype=np.float32) * PL_SCALE_DB + PL_OFFSET_DB


def augment_code(seed, epoch, sample_id, band):
    """Return a stable D4 code, independent of Python/NumPy/worker RNG state."""
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("seed/epoch must be nonnegative integers")
    data = json.dumps([seed, epoch, str(sample_id), str(band)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") % 8


def spatial_transform(array, code):
    """Rotate on the last two axes, then optionally flip columns; always copy."""
    if type(code) is not int or not 0 <= code < 8:
        raise ValueError("spatial transform code must be an integer in [0,7]")
    result = np.rot90(array, code % 4, axes=(-2, -1))
    if code >= 4:
        result = np.flip(result, axis=-1)
    return np.array(result, copy=True, order="C")
