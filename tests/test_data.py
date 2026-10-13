from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lunar_rm.data import LunarDataset
from tests.fixtures.official import build_dataset


def official_class(path):
    module = types.ModuleType("official_reference_for_test")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module.LunarRadioMapDataset


@pytest.mark.parametrize("nested", [False, True])
def test_raw_sample_contract_preserves_identity_and_gt(tmp_path, nested):
    root = build_dataset(tmp_path / "data", nested=nested)
    ds = LunarDataset(root)
    assert len(ds) == 6
    first, boundary, last = ds[0], ds[3], ds[-1]
    assert first["sample_id"] == boundary["sample_id"] == "00001_00"
    assert (first["band"], boundary["band"], last["band"]) == ("415", "58", "58")
    assert first["terrain_id"] == "00001" and first["hm_row"] == 1
    assert first["raw_dem_m"][0, 0, 0] == 5  # label row 0 references DEM row 1
    assert first["tx_map"][0, 0, 255] == 1 and first["tx_map"].sum() == 1
    assert first["target_db"][0, 0, :3].tolist() == [145, 10, 240]
    assert first["valid_mask"][0, 0, 0]  # 145 dB can be valid
    assert not first["valid_mask"][0, 1, 0]
    assert not torch.equal(first["target_db"], boundary["target_db"])
    assert first["features"].shape == (3, 256, 256)
    torch.testing.assert_close(first["features"][0], first["raw_dem_m"][0] / 496)
    assert boundary["features"][2].eq(1).all()
    assert first["valid_mask"].dtype == torch.bool
    assert all(isinstance(a, np.memmap) and not a.flags.writeable for a in ds._arrays.values())
    original = ds[0]["target_db"].clone()
    first["target_db"].zero_()
    torch.testing.assert_close(ds[0]["target_db"], original)
    assert ds.identity["files"]["metadata.json"]["sha256"]


@pytest.mark.parametrize("features", [("height", "valid_mask"), ("target_db",), ("tx", "tx")])
def test_label_inputs_and_duplicate_features_rejected(tmp_path, features):
    with pytest.raises(ValueError):
        LunarDataset(build_dataset(tmp_path / "data"), features=features)


def test_missing_test_and_invalid_index(tmp_path):
    root = build_dataset(tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="test"):
        LunarDataset(root, split="test")
    ds = LunarDataset(root, bands=("58",))
    assert len(ds) == 3 and ds[0]["band"] == "58"
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-len(ds) - 1]


def test_spawn_workers_equal_single_process_after_cache_open(tmp_path, monkeypatch):
    # Conda's NumPy can set INTEL in the parent while PyTorch loads libgomp;
    # spawned interpreters must use the compatible GNU threading backend.
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    ds = LunarDataset(build_dataset(tmp_path / "data"))
    expected = [ds[i] for i in range(len(ds))]
    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context="spawn", timeout=30)
    seen = []
    for batch in loader:
        for j, sid in enumerate(batch["sample_id"]):
            k = len(seen)
            assert (sid, batch["band"][j]) == (expected[k]["sample_id"], expected[k]["band"])
            for key in ("raw_dem_m", "tx_map", "target_db", "valid_mask", "features"):
                torch.testing.assert_close(batch[key][j], expected[k][key])
            seen.append((sid, batch["band"][j]))
    assert len(set(seen)) == len(ds)


def test_full_feature_cache_matches_uncached_and_los_is_not_mask(tmp_path):
    from lunar_rm.registry import FULL_FEATURES
    root = build_dataset(tmp_path / "data")
    plain = LunarDataset(root, features=FULL_FEATURES)[0]
    ds = LunarDataset(root, features=FULL_FEATURES, cache_dir=tmp_path / "cache")
    cold, warm = ds[0], ds[0]
    for key in ("features", "baseline_db", "geometric_los"):
        torch.testing.assert_close(plain[key], cold[key], atol=0, rtol=0)
        torch.testing.assert_close(cold[key], warm[key], atol=0, rtol=0)
    # The synthetic DEM is planar: invalid supervision is still geometric LoS.
    assert plain["geometric_los"].all()
    assert not plain["valid_mask"].all()
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1


@pytest.mark.real_data
@pytest.mark.parametrize("split", ["train", "val"])
def test_official_equivalence_at_both_band_boundaries(split):
    root = Path("/home/DataDisk/zsfang/lunarRM")
    if not (root / "lunar_dataset.py").exists():
        pytest.skip("official loader/data not installed")
    official = official_class(root / "lunar_dataset.py")(
        root, split=split, band="both", normalize=False, augment=False,
        return_name=True, return_mask=True)
    ds = LunarDataset(root, split=split)
    assert len(ds) == len(official)
    n = len(ds) // 2
    for index in (0, n - 2, n - 1, n, n + 1, 2 * n - 1):
        x, target, mask, name = official[index]
        actual = ds[index]
        torch.testing.assert_close(actual["raw_dem_m"], x[0:1], rtol=0, atol=0)
        torch.testing.assert_close(actual["tx_map"], x[1:2], rtol=0, atol=0)
        torch.testing.assert_close(actual["target_db"], target, rtol=0, atol=0)
        assert torch.equal(actual["valid_mask"], mask.bool())
        assert name == f"{actual['sample_id']}_{actual['band']}"
