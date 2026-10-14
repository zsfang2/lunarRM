"""Metre/Hz/dB physical features; DEM and TX use stored raster coordinates."""

import math

import numpy as np


SPEED_OF_LIGHT_M_S = 299792458.0
BAND_FREQUENCIES_HZ = {"415": 415e6, "58": 5775e6}
FEATURE_VERSION = "bilinear-interior-slant-fresnel-v1"


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


def geometry_features(dem_m, tx, *, spacing_m=1.0, dtype=np.float64):
    """Relative coordinates/heights in m, slope dimensionless, Laplacian in 1/m.

    The Laplacian is a curvature proxy, not strict surface curvature. Derivatives
    use central differences internally and one-sided differences at boundaries.
    """
    dem, (r, c), dtype = _geometry(dem_m, tx, spacing_m, dtype)
    rr, cc = np.indices(dem.shape, dtype=dtype)
    def derivative(a, axis):
        return np.gradient(a, spacing_m, axis=axis) if a.shape[axis] > 1 else np.zeros_like(a)
    dr, dc = derivative(dem, 0), derivative(dem, 1)
    return {"relative_row": (rr - r) * spacing_m, "relative_col": (cc - c) * spacing_m,
            "relative_height": dem - dem[r, c], "slope": np.hypot(dr, dc),
            "curvature_proxy": derivative(dr, 0) + derivative(dc, 1)}


def _bilinear(dem, rows, cols):
    rows, cols = np.clip(rows, 0, dem.shape[0]-1), np.clip(cols, 0, dem.shape[1]-1)
    r0, c0 = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    r1, c1 = np.minimum(r0+1, dem.shape[0]-1), np.minimum(c0+1, dem.shape[1]-1)
    wr, wc = rows-r0, cols-c0
    return (dem[r0, c0]*(1-wr)*(1-wc) + dem[r1, c0]*wr*(1-wc)
            + dem[r0, c1]*(1-wr)*wc + dem[r1, c1]*wr*wc)


def profile_features(dem_m, tx, frequency_hz, *, spacing_m=1.0, step_m=1.0,
                     chunk_size=128, memory_budget_mb=64, dtype=np.float64):
    """Bilinear terrain profiles with interior samples at horizontal spacing <= step.

    Clearance is antenna-to-antenna line height minus terrain height. Nonnegative
    minimum clearance means LoS. Fresnel uses sqrt(lambda * D * t * (1-t)), where
    D is the slant Tx/Rx distance and t is the horizontal path fraction. Zero or
    unsampled short horizontal paths are LoS with clearance/fresnel set to zero.
    Receiver chunks bound temporary matrices; an estimated 32 arrays per point
    are budgeted before any profile matrix is allocated.
    """
    dem, (tr, tc), dtype = _geometry(dem_m, tx, spacing_m, dtype)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency must be positive finite Hz")
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError("profile step must be positive finite metres")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not np.isfinite(memory_budget_mb) or memory_budget_mb <= 0:
        raise ValueError("memory budget must be positive")
    diagonal = np.hypot(max(tr, dem.shape[0]-1-tr), max(tc, dem.shape[1]-1-tc))*spacing_m
    max_steps = max(1, math.ceil(diagonal / step_m))
    # _bilinear indexing uses int64 even for float32 input, so budget at least 8 bytes.
    bytes_per_receiver = max_steps * 32 * max(8, dtype.itemsize)
    safe_chunk = min(chunk_size, int(memory_budget_mb * 1024**2 // bytes_per_receiver))
    if safe_chunk < 1:
        raise ValueError("profile sampling exceeds temporary memory budget")
    clearance = np.zeros(dem.size, dtype=dtype)
    fresnel = np.zeros(dem.size, dtype=dtype)
    tx_height = dem[tr, tc] + 3
    slant = distance_3d(dem, (tr, tc), spacing_m=spacing_m, dtype=dtype).ravel()
    for start in range(0, dem.size, safe_chunk):
        indices = np.arange(start, min(start + safe_chunk, dem.size))
        rr, cc = np.unravel_index(indices, dem.shape)
        horizontal = np.hypot(rr-tr, cc-tc) * spacing_m
        steps = np.maximum(1, np.ceil(horizontal / step_m).astype(np.int64))
        if steps.max() <= 1:
            continue
        j = np.arange(1, steps.max(), dtype=dtype)[None, :]
        valid = j < steps[:, None]
        t = np.minimum(j / steps[:, None], 1)
        terrain = _bilinear(dem, tr+(rr-tr)[:, None]*t, tc+(cc-tc)[:, None]*t)
        line = tx_height + (dem[rr, cc]+1-tx_height)[:, None]*t
        gap = line-terrain
        radius = np.sqrt((SPEED_OF_LIGHT_M_S/frequency_hz)*slant[indices, None]*t*(1-t))
        ratio = np.full_like(gap, np.inf)
        np.divide(gap, radius, out=ratio, where=valid)
        gap = np.where(valid, gap, np.inf)
        minimum_gap, minimum_ratio = gap.min(axis=1), ratio.min(axis=1)
        no_interior = steps <= 1
        clearance[indices] = np.where(no_interior, 0, minimum_gap)
        fresnel[indices] = np.where(no_interior, 0, minimum_ratio)
    return {"los": (clearance >= 0).reshape(dem.shape),
            "clearance_m": clearance.reshape(dem.shape), "fresnel": fresnel.reshape(dem.shape)}
