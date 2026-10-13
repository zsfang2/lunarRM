import pytest
import torch
from torch import nn

from lunar_rm.config import resolve_config
from lunar_rm.models import build_model, parameter_report
from lunar_rm.registry import EXPERIMENTS


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("experiment", list(EXPERIMENTS))
def test_full_resolution_forward_and_parameter_reporting(experiment):
    cfg = resolve_config(experiment=experiment)
    model = build_model(cfg).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, len(cfg.features), 256, 256), torch.ones(1, 1, 256, 256) * 90)
    assert result.shape == (1, 1, 256, 256) and torch.isfinite(result).all()
    report = parameter_report(model)
    assert report["parameters"] == sum(p.numel() for p in model.parameters())
    if cfg.model == "tiny":
        assert 100000 <= report["parameters"] <= 500000
        assert report["tiny_target_met"]
        assert any(isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.kernel_size == (3, 3)
                   for m in model.modules())
    else:
        assert report["tiny_target_met"] is None


def test_signed_residual_zero_baseline_and_direct_intercept():
    for name in ("R0", "G0"):
        model = build_model(resolve_config(experiment=name))
        for p in model.parameters():
            nn.init.zeros_(p)
        x, base = torch.zeros(1, 4, 32, 32), torch.ones(1, 1, 32, 32) * 80
        for bias in (0.0, -.25, .25):
            model.backbone.head.bias.data.fill_(bias)
            pred = model(x, base)
            expected = (80 if name == "R0" else 20) + 208 * bias
            torch.testing.assert_close(pred, torch.full_like(pred, expected))
    r = build_model(resolve_config(experiment="R0"))
    g = build_model(resolve_config(experiment="G0"))
    assert parameter_report(r)["parameters"] == parameter_report(g)["parameters"]
    with pytest.raises(ValueError, match="baseline"):
        r(torch.zeros(1, 4, 32, 32))
    with pytest.raises(ValueError, match="channels"):
        r(torch.zeros(1, 3, 32, 32), torch.zeros(1, 1, 32, 32))


def test_checkpoint_rejects_same_shape_different_feature_order():
    cfg = resolve_config(experiment="R0")
    a = build_model(cfg)
    b = build_model(resolve_config(experiment="R0", overrides={"features": list(reversed(cfg.features))}))
    with pytest.raises(ValueError, match="signature"):
        b.load_state_dict(a.state_dict())
