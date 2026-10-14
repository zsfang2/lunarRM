import math

import pytest
import torch

from lunar_rm.reporting import GroupedMetrics, fixed_indices
from lunar_rm.config import resolve_config
from lunar_rm.data import LunarDataset
from tests.fixtures.official import build_dataset


def test_groups_use_valid_intersection_and_terrain_equal():
    m = GroupedMetrics((0.5, 0.9))
    pred = torch.tensor([[[[2., 900.]]], [[[4., 6.]]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[True, False]]], [[[True, True]]]])
    los = torch.tensor([[[[True, False]]], [[[False, False]]]])
    m.update(pred, target, mask, ["415", "415"], los, ["single", "multi"], ["a", "b"])
    report = m.report()
    assert report["groups"]["los"]["bands"]["415"]["rmse_db"] == 2
    assert report["groups"]["nlos"]["bands"]["415"]["rmse_db"] == pytest.approx(math.sqrt(26))
    assert report["groups"]["single"]["bands"]["415"]["valid_pixels"] == 1
    assert report["groups"]["coverage_0"]["bands"]["415"]["items"] == 0
    assert report["groups"]["coverage_1"]["bands"]["415"]["items"] == 1
    assert report["terrain_equal"]["415"]["mean_terrain_rmse_db"] == pytest.approx((2+math.sqrt(26))/2)
    assert report["terrain_equal"]["58"]["mean_terrain_rmse_db"] is None
    assert report["groups"]["single"]["bands"]["58"]["rmse_db"] is None
    with pytest.raises(ValueError):
        m.update(pred, target, mask, ["415", "415"], los.float(), ["single", "multi"], ["a", "b"])
    with pytest.raises(ValueError):
        GroupedMetrics((0.9, 0.5))
    separated = GroupedMetrics((0.5, 0.9))
    for i, group in enumerate(["single", "multi"]):
        separated.update(pred[i:i+1], target[i:i+1], mask[i:i+1], ["415"],
                         los[i:i+1], [group], [["a", "b"][i]])
    assert separated.report() == report


def test_grouped_evaluation_does_not_change_global_metrics(tmp_path):
    from lunar_rm.evaluation import evaluate_model
    from lunar_rm.models import build_model
    ds = LunarDataset(build_dataset(tmp_path / "data"), split="val", features=(), need_los=True)
    model = build_model(resolve_config(experiment="A0"))
    base = evaluate_model(model, ds)
    diagnostic = evaluate_model(model, ds, grouped=True)
    assert {k: v for k, v in diagnostic.items() if k != "diagnostics"} == base
    for band in ("415", "58"):
        groups = diagnostic["diagnostics"]["groups"]
        assert sum(groups[k]["bands"][band]["valid_pixels"] for k in ("los", "nlos")) == base["bands"][band]["valid_pixels"]


def test_fixed_examples_and_ablation_fairness(tmp_path):
    ds = LunarDataset(build_dataset(tmp_path / "data"), split="val")
    ids = [{"band": "58", "sample_id": ds.rows[0]["sample_id"]},
           {"band": "415", "sample_id": ds.rows[1]["sample_id"]}]
    assert fixed_indices(ds, ids) == [2, 1]
    with pytest.raises(ValueError, match="duplicate"):
        fixed_indices(ds, ids+ids)
    with pytest.raises(ValueError, match="unknown"):
        fixed_indices(ds, [{"band": "415", "sample_id": "missing"}])
    base = resolve_config(experiment="R2").to_dict()
    expected = {"no_los": {"features"}, "no_fresnel": {"features"},
                "no_curvature": {"features"}, "mse": {"loss"}, "mae": set(),
                "alpha0": set(), "alpha01": {"nlos_alpha"}, "alpha05": {"nlos_alpha"},
                "direct": {"head"}}
    for suffix, changed in expected.items():
        other = resolve_config(config_path=f"configs/R2_{suffix}.json").to_dict()
        diff = {k for k in base if base[k] != other[k]} - {"experiment", "output_dir"}
        assert diff == changed


def test_diagnostics_cli_and_fixed_panels(tmp_path):
    import json
    import subprocess
    import sys
    from lunar_rm.config import PROJECT_ROOT
    root = build_dataset(tmp_path / "data")
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([{"band": "415", "sample_id": "00010_00"}]))
    out = tmp_path / "diagnostics"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/diagnose.py"), "--experiment", "A0",
           "--data-root", str(root), "--device", "cpu", "--max-per-band", "1",
           "--examples", str(examples), "--output-dir", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    report = json.loads((out / "diagnostics.json").read_text())
    assert report["metrics"]["scope"] == "subset"
    assert report["metrics"]["items_seen"] == 2
    assert report["examples"] == [{"band": "415", "sample_id": "00010_00"}]
    assert (out / "example-000.png").is_file()
    assert (out / "example-000.npz").is_file()
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
