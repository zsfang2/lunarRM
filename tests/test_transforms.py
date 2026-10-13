import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lunar_rm.data import LunarDataset
from lunar_rm.features import fspl_db
from lunar_rm.transforms import (augment_code, denormalize_pathloss, normalize_height,
                                 normalize_pathloss, spatial_transform)
from tests.fixtures.official import build_dataset


def test_public_normalization_keeps_unclipped_gt():
    dem = np.array([-1, 0, 248, 496, 500], dtype=np.float32)
    np.testing.assert_array_equal(normalize_height(dem), [0, 0, .5, 1, 1])
    gt = np.array([10, 20, 124, 228, 240], dtype=np.float32)
    before = gt.copy()
    normalized = normalize_pathloss(gt)
    assert normalized[0] < 0 and normalized[-1] > 1
    np.testing.assert_allclose(denormalize_pathloss(normalized), gt, atol=1e-5)
    np.testing.assert_array_equal(normalize_pathloss(gt, clip=True), [0, 0, .5, 1, 1])
    np.testing.assert_array_equal(gt, before)


@pytest.mark.parametrize("code", range(8))
def test_spatial_transform_known_tx_positions_and_alignment(code):
    # rot90 followed by an optional horizontal flip, independently tabulated.
    positions = [(0, 2), (1, 0), (3, 1), (2, 3),
                 (0, 1), (1, 3), (3, 2), (2, 0)]
    tx = np.zeros((4, 4), dtype=np.float32)
    tx[0, 2] = 1
    transformed = spatial_transform(tx, code)
    assert tuple(np.argwhere(transformed == 1)[0]) == positions[code]
    np.testing.assert_array_equal(spatial_transform(tx * 7, code), transformed * 7)
    assert transformed.flags.c_contiguous


def test_stable_seed_and_independent_raw_gt_in_dataset(tmp_path):
    root = build_dataset(tmp_path / "data")
    ds = LunarDataset(root, augment=True, seed=2027, target_mode="official_clipped")
    first = ds[0]
    np.random.seed(37)
    again = ds[0]
    for key in ("raw_dem_m", "target_db", "valid_mask", "tx_map", "features"):
        torch.testing.assert_close(first[key], again[key])
    assert first["target_db"].min() == 10 and first["target_db"].max() == 240
    assert first["training_target_db"].min() == 20 and first["training_target_db"].max() == 228
    raw = LunarDataset(root)[0]
    code = augment_code(2027, 0, raw["sample_id"], raw["band"])
    for key in ("raw_dem_m", "target_db", "valid_mask", "tx_map"):
        np.testing.assert_array_equal(first[key].numpy(), spatial_transform(raw[key].numpy(), code))
    r, c = first["tx_row"], first["tx_col"]
    assert first["tx_map"][0, r, c] == 1
    assert (first["original_tx_row"], first["original_tx_col"]) == (0, 255)
    assert first["clipped_valid_pixels"] == 2
    assert augment_code(2027, 0, raw["sample_id"], "415") == code


def test_val_augmentation_rejected(tmp_path):
    with pytest.raises(ValueError, match="train"):
        LunarDataset(build_dataset(tmp_path / "data"), split="val", augment=True)


def test_physics_uses_transformed_raw_metres_and_never_labels(tmp_path):
    root = build_dataset(tmp_path / "data")
    ds = LunarDataset(root, features=("height", "tx", "band", "fspl"), augment=True)
    sample = ds[0]
    expected = fspl_db(sample["raw_dem_m"][0].numpy(),
                       (sample["tx_row"], sample["tx_col"]), 415e6, dtype=np.float32)
    np.testing.assert_allclose(sample["baseline_db"][0], expected, atol=1e-4)
    np.testing.assert_allclose(sample["features"][3], normalize_pathloss(expected), atol=1e-6)
    # Only synthetic targets are changed; features must be independent of them.
    targets = np.load(root / "train_pl415.npy")
    targets.fill(123)
    np.save(root / "train_pl415.npy", targets)
    changed = LunarDataset(root, features=ds.feature_names, augment=True)[0]
    torch.testing.assert_close(changed["features"], sample["features"])
    assert not torch.equal(changed["target_db"], sample["target_db"])


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_invalid_epoch_rejected(tmp_path, epoch):
    ds = LunarDataset(build_dataset(tmp_path / "data"))
    with pytest.raises(ValueError):
        ds.set_epoch(epoch)


def test_epoch_shared_with_persistent_spawn_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    ds = LunarDataset(build_dataset(tmp_path / "data"), augment=True, seed=3)
    loader = DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True,
                        multiprocessing_context="spawn", timeout=30)
    observed_codes = []
    try:
        for epoch in (0, 2, 0):
            ds.set_epoch(epoch)
            codes = []
            for b, batch in enumerate(loader):
                for j in range(len(batch["sample_id"])):
                    expected = ds[b * 2 + j]
                    for key in ("target_db", "raw_dem_m", "features", "valid_mask"):
                        torch.testing.assert_close(batch[key][j], expected[key])
                    codes.append(int(batch["augmentation_code"][j]))
            observed_codes.append(codes)
    finally:
        del loader
    assert observed_codes[0] == observed_codes[2]
    assert observed_codes[0] != observed_codes[1]
