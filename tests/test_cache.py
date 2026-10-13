import numpy as np
import pytest

from lunar_rm.cache import FeatureCache, cache_key


def test_cache_roundtrip_dependencies_and_incomplete_rejection(tmp_path):
    cache = FeatureCache(tmp_path / "cache", data_root=tmp_path / "data")
    identity = {"data": "fixture", "version": "v1", "frequency_hz": 415e6, "dtype": "float32"}
    key = cache_key(identity)
    arrays = {"fspl": np.ones((3, 3), dtype="float32"), "los": np.ones((3, 3), dtype=bool)}
    assert cache.get(key, identity, {"fspl", "los"}, (3, 3)) is None
    cache.put(key, identity, arrays)
    actual = cache.get(key, identity, set(arrays), (3, 3))
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], actual[name])
    for changed in ({**identity, "version": "v2"}, {**identity, "frequency_hz": 5775e6}):
        assert cache_key(changed) != key
        assert cache.get(cache_key(changed), changed, set(arrays), (3, 3)) is None
    with pytest.raises(ValueError, match="identity"):
        cache.get(key, {**identity, "version": "v2"}, set(arrays), (3, 3))
    with pytest.raises(ValueError, match="fields"):
        cache.get(key, identity, {"fspl", "los", "fresnel"}, (3, 3))
    with pytest.raises(ValueError, match="data root"):
        FeatureCache(tmp_path / "data" / "cache", data_root=tmp_path / "data")
