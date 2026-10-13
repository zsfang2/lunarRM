import numpy as np
import pytest

from lunar_rm.features import geometry_features, profile_features


def test_geometry_units_and_flat_profiles():
    rr, cc = np.indices((5, 5))
    geom = geometry_features((2 * rr + 3 * cc).astype(float), (2, 1), spacing_m=2)
    assert geom["relative_row"][3, 1] == 2
    assert geom["relative_col"][2, 3] == 4
    assert geom["relative_height"][3, 1] == 2
    assert geom["slope"][2, 2] == pytest.approx(np.hypot(1, 1.5))
    assert geom["curvature_proxy"][2, 2] == pytest.approx(0)
    result = profile_features(np.zeros((5, 5)), (0, 0), 415e6)
    assert result["los"].all()
    assert result["fresnel"][0, 0] == result["clearance_m"][0, 0] == 0
    assert np.isfinite(result["fresnel"]).all()


@pytest.mark.parametrize("step", [.5, 1, 2])
def test_ridge_blocks_los_frequency_changes_only_fresnel(step):
    dem = np.zeros((5, 5)); dem[:, 2] = 10
    low = profile_features(dem, (2, 0), 415e6, step_m=step)
    high = profile_features(dem, (2, 0), 5775e6, step_m=step)
    assert not low["los"][2, 4]
    assert low["clearance_m"][2, 4] < 0
    np.testing.assert_array_equal(low["los"], high["los"])
    np.testing.assert_allclose(low["clearance_m"], high["clearance_m"])
    assert high["fresnel"][2, 4] == pytest.approx(low["fresnel"][2, 4] * np.sqrt(5775/415))


def test_bilinear_interior_samples_and_chunk_invariance(monkeypatch):
    import lunar_rm.features as module
    dem = np.array([[0, 0, 0], [0, 8, 0], [0, 0, 0]], dtype=float)
    sizes = []
    original = module._bilinear
    def checked(dem, rows, cols):
        sizes.append(rows.shape[0])
        return original(dem, rows, cols)
    monkeypatch.setattr(module, "_bilinear", checked)
    a = profile_features(dem, (0, 0), 415e6, chunk_size=2, step_m=.5)
    assert max(sizes) <= 2
    b = profile_features(dem, (0, 0), 415e6, chunk_size=32, step_m=.5)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)
    assert not a["los"][2, 2]
    assert original(dem, np.array([[.5]]), np.array([[.5]]))[0, 0] == 2


@pytest.mark.parametrize("kw", [{"step_m": 0}, {"chunk_size": 0},
                                {"step_m": 1e-10}, {"memory_budget_mb": .00001}])
def test_profile_rejects_bad_steps_or_unbounded_memory(kw):
    with pytest.raises(ValueError):
        profile_features(np.zeros((5, 5)), (0, 0), 415e6, **kw)
