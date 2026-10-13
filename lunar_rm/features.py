"""Metre/Hz/dB physical features; DEM and TX use stored raster coordinates."""

import math

import numpy as np


SPEED_OF_LIGHT_M_S = 299792458.0
BAND_FREQUENCIES_HZ = {"415": 415e6, "58": 5775e6}


def _geometry(dem_m, tx, spacing_m, dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("physical feature dtype must be float32 or float64")
    dem = np.asarray(dem_m, dtype=dtype)
    if dem.ndim != 2 or min(dem.shape) < 1 or not np.isfinite(dem).all():
        raise ValueError("DEM must be a nonempty finite 2D metre array")
    if isinstance(spacing_m, bool) or not np.isscalar(spacing_m) or not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("grid spacing must be positive finite metres")
    if len(tx) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in tx):
        raise ValueError("TX must contain two integer raster coordinates")
    r, c = tx
    if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
        raise ValueError("TX is outside DEM bounds")
    return dem, (int(r), int(c)), dtype


def distance_3d(dem_m, tx, *, spacing_m=1.0, dtype=np.float64):
    """Distance in metres: Tx ground+3m to each Rx ground+1m, including TX pixel."""
    dem, (r, c), dtype = _geometry(dem_m, tx, spacing_m, dtype)
    rr, cc = np.indices(dem.shape, dtype=dtype)
    horizontal = np.hypot((rr - r) * spacing_m, (cc - c) * spacing_m)
    vertical = (dem - dem[r, c]) - 2.0
    distance = np.hypot(horizontal, vertical).astype(dtype)
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("distance is nonfinite or nonpositive in requested precision")
    return distance


def fspl_db(dem_m, tx, frequency_hz, *, spacing_m=1.0, dtype=np.float64):
    """Free-space path loss, 20 log10(4 pi d f / c), with no power offset."""
    if isinstance(frequency_hz, bool) or not np.isscalar(frequency_hz) or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency must be positive finite Hz")
    distance = distance_3d(dem_m, tx, spacing_m=spacing_m, dtype=dtype)
    # Sum logarithms to avoid overflow of the distance-frequency product.
    result = 20 * np.log10(distance) + 20 * math.log10(frequency_hz) + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT_M_S)
    return result.astype(dtype)
