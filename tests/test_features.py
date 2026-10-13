import math

import numpy as np
import pytest

from lunar_rm.features import distance_3d, fspl_db
from lunar_rm.transforms import spatial_transform


@pytest.mark.parametrize("dtype, atol", [(np.float64, 1e-6), (np.float32, 1e-4)])
def test_analytic_distance_fspl_and_frequency_difference(dtype, atol):
    dem = np.zeros((5, 5), dtype=dtype)
    d = distance_3d(dem, (0, 0), spacing_m=1, dtype=dtype)
    assert d[0, 0] == 2
    assert d[0, 3] == pytest.approx(math.sqrt(13), abs=1e-6)
    low = fspl_db(dem, (0, 0), 415e6, dtype=dtype)
    high = fspl_db(dem, (0, 0), 5775e6, dtype=dtype)
    expected = 20 * math.log10(4 * math.pi * math.sqrt(13) * 415e6 / 299792458)
    assert low[0, 3] == pytest.approx(expected, abs=atol)
    np.testing.assert_allclose(high - low, 20 * math.log10(5775 / 415), rtol=0, atol=atol)
    np.testing.assert_allclose(fspl_db(dem + 1024, (0, 0), 415e6, dtype=dtype), low, rtol=0, atol=atol)
    assert low.dtype == np.dtype(dtype)


def test_asymmetric_elevation_no_extra_flip_and_metre_spacing():
    dem = np.array([[0, 7, 1], [3, 4, 8]], dtype=np.float64)
    actual = distance_3d(dem, (0, 1), spacing_m=2)
    expected = math.sqrt((2 * 1)**2 + (2 * 1)**2 + ((7 + 3) - (8 + 1))**2)
    assert actual[1, 2] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"frequency_hz": 0}, {"frequency_hz": -1}, {"frequency_hz": float("nan")},
    {"spacing_m": 0}, {"spacing_m": -1}, {"spacing_m": float("inf")},
    {"tx": (-1, 0)}, {"tx": (0, 3)}, {"tx": (0.5, 1)},
    {"dtype": "float16"},
])
def test_reject_bad_physical_inputs(kwargs):
    args = {"dem_m": np.zeros((3, 3)), "tx": (0, 0), "frequency_hz": 415e6, **kwargs}
    with pytest.raises(ValueError):
        fspl_db(**args)


def test_reject_nonfinite_and_nonmap_dem():
    for bad in (np.array([[np.nan]]), np.array([[np.inf]]), np.zeros(4)):
        with pytest.raises(ValueError):
            fspl_db(bad, (0, 0), 415e6)


@pytest.mark.parametrize("code", range(8))
def test_fspl_spatial_equivariance(code):
    dem = np.arange(16).reshape(4, 4) * 0.5
    tx = np.zeros((4, 4)); tx[0, 2] = 1
    r, c = np.argwhere(spatial_transform(tx, code) == 1)[0]
    transformed = fspl_db(spatial_transform(dem, code), (int(r), int(c)), 415e6)
    np.testing.assert_allclose(transformed, spatial_transform(fspl_db(dem, (0, 2), 415e6), code), atol=1e-10)
