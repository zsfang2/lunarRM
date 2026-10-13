import pytest
import torch

from lunar_rm.losses import masked_band_loss
from lunar_rm.sampling import BalancedBatchSampler


@pytest.mark.parametrize("kind, expected", [("mae", 2), ("mse", 5)])
def test_equal_band_pixel_means_and_invalid_gradients(kind, expected):
    pred = torch.tensor([[[[1., 1., 1., 1.]]], [[[3., 999., 999., 999.]]]], requires_grad=True)
    mask = torch.tensor([[[[1, 1, 1, 1]]], [[[1, 0, 0, 0]]]], dtype=torch.bool)
    result = masked_band_loss(pred, torch.zeros_like(pred), mask, ["415", "58"], kind=kind)
    assert result["error"].item() == expected
    assert result["loss"].dtype == torch.float32
    assert result["loss"].item() == pytest.approx(expected / (208 if kind == "mae" else 208**2))
    result["loss"].backward()
    assert pred.grad[1, 0, 0, 1:].eq(0).all()


def test_empty_and_absent_band_are_not_zero_loss_terms():
    pred = torch.ones(2, 1, 2, 2, requires_grad=True) * 4
    mask = torch.zeros_like(pred, dtype=torch.bool); mask[0] = True
    r = masked_band_loss(pred, torch.zeros_like(pred), mask, ["415", "58"])
    assert r["error"].item() == 4 and r["skipped_items"] == 1
    mask.zero_()
    assert masked_band_loss(pred, pred, mask, ["415", "58"])["loss"] is None


def test_nlos_addition_uses_only_valid_nlos_and_ignores_invalid_nan():
    pred = torch.tensor([[[[2., 4., float("nan")]]]], requires_grad=True)
    mask = torch.tensor([[[[True, True, False]]]])
    los = torch.tensor([[[[True, False, False]]]])
    result = masked_band_loss(pred, torch.zeros_like(pred), mask, ["415"],
                              geometric_los=los, nlos_alpha=.5)
    assert result["error"].item() == 5  # mean(2,4) + .5 * 4
    result["loss"].backward()
    assert pred.grad[0, 0, 0, 2] == 0
    with pytest.raises(ValueError, match="LoS"):
        masked_band_loss(pred, pred, mask, ["415"], nlos_alpha=.1)
    with pytest.raises(FloatingPointError):
        masked_band_loss(pred, torch.zeros_like(pred), torch.ones_like(mask), ["415"])


def test_balanced_sampler_covers_bands_and_keeps_short_balanced_tail():
    sampler = BalancedBatchSampler(5, 2, 4, seed=3)
    batches = list(sampler)
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(i for b in batches for i in b) == list(range(10))
    for batch in batches:
        assert sum(i < 5 for i in batch) == len(batch)//2
    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches
    restored = BalancedBatchSampler(5, 2, 4, seed=3)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == list(sampler)
    with pytest.raises(ValueError):
        BalancedBatchSampler(5, 2, 3)
