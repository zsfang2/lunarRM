import math

import pytest
import torch

from lunar_rm.metrics import BandMetrics, selection_score
from lunar_rm.evaluation import evaluate_model, evaluation_indices
from lunar_rm.data import LunarDataset
from lunar_rm.config import resolve_config
from lunar_rm.models import build_model
from tests.fixtures.official import build_dataset


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_float64_global_accumulation_and_missing_band():
    m = BandMetrics()
    pred = torch.tensor([[[[1., 3.]]], [[[5., 999.]]]], dtype=torch.float32)
    mask = torch.tensor([[[[1, 1]]], [[[1, 0]]]], dtype=torch.bool)
    m.update(pred, torch.zeros_like(pred), mask, ["415", "415"])
    report = m.report()
    assert report["bands"]["415"]["sse"] == 35
    assert report["bands"]["415"]["rmse_db"] == pytest.approx(math.sqrt(35/3))
    assert report["bands"]["415"]["mae_db"] == 3
    assert report["equal_band_rmse_db"] is None
    assert report["bands"]["58"]["rmse_db"] is None
    n = BandMetrics()
    for i in range(2):
        n.update(pred[i:i+1], torch.zeros_like(pred[i:i+1]), mask[i:i+1], ["415"])
    assert n.report() == report


def test_complete_and_fixed_subset_evaluation_batch_invariance(tmp_path):
    ds = LunarDataset(build_dataset(tmp_path / "data"), split="val", features=())
    model = build_model(resolve_config(experiment="A0"))
    results = [evaluate_model(model, ds, batch_size=b) for b in (1, 3, 4)]
    assert results[0] == results[1] == results[2]
    full = results[0]
    assert full["scope"] == "full" and full["items_seen"] == full["items_expected"] == 4
    assert selection_score(full) == full["equal_band_rmse_db"]
    assert evaluation_indices(ds, max_per_band=1) == [0, 2]
    limited = evaluate_model(model, ds, max_per_band=1)
    assert limited["scope"] == "subset" and limited["items_seen"] == 2
    with pytest.raises(ValueError, match="full"):
        selection_score(limited)
    # Raw out-of-range GT is evaluated; changing only training_target_db must not affect metrics.
    clipped = LunarDataset(ds.root, split="val", features=(), target_mode="official_clipped")
    assert evaluate_model(model, clipped) == full


def test_evaluator_detects_duplicate_and_empty_bands(tmp_path):
    ds = LunarDataset(build_dataset(tmp_path / "data"), split="val", features=())
    ds.rows[1]["sample_id"] = ds.rows[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_model(build_model(resolve_config(experiment="A0")), ds)
    m = BandMetrics()
    m.update(torch.zeros(1,1,1,1), torch.zeros(1,1,1,1), torch.zeros(1,1,1,1,dtype=torch.bool), ["415"])
    assert m.report()["bands"]["415"]["empty_items"] == 1


def test_checkpoint_evaluation_and_identity_rejection(tmp_path):
    from lunar_rm.evaluation import evaluate_checkpoint
    from lunar_rm.data import dataset_from_config
    cfg = resolve_config(experiment="N0", overrides={"data_root": str(build_dataset(tmp_path / "data")),
                         "device": "cpu", "width": 4, "groups": 2, "num_workers": 0})
    ds = dataset_from_config(cfg, "val")
    model = build_model(cfg)
    payload = {"format_version": 1, "model": model.state_dict(), "config": cfg.to_dict(),
               "data_identity": {"val": ds.identity}}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    report = evaluate_checkpoint(cfg, path, batch_size=2)
    assert report["metrics"] == evaluate_model(model, ds, batch_size=2)
    assert report["checkpoint"]["sha256"]
    payload["data_identity"]["val"] = {}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="data identity"):
        evaluate_checkpoint(cfg, path)
    payload["data_identity"]["val"] = ds.identity
    payload["config"]["features"] = ["tx", "height", "band"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="config"):
        evaluate_checkpoint(cfg, path)
    path.write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_checkpoint(cfg, path)
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_checkpoint(cfg)


def test_metric_invalid_nan_ignored_valid_nan_rejected():
    pred = torch.tensor([[[[1., float("nan")]]]], dtype=torch.float64)
    m = BandMetrics()
    mask = torch.tensor([[[[True, False]]]])
    m.update(pred, torch.zeros_like(pred), mask, ["415"])
    assert m.report()["bands"]["415"]["sse"] == 1
    with pytest.raises(FloatingPointError):
        m.update(pred, torch.zeros_like(pred), ~mask, ["415"])


def test_evaluation_cli_synthetic_a0(tmp_path):
    import json
    import subprocess
    import sys
    from lunar_rm.config import PROJECT_ROOT
    root = build_dataset(tmp_path / "data")
    output = tmp_path / "metrics.json"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"), "--experiment", "A0",
           "--data-root", str(root), "--device", "cpu", "--num-workers", "0", "--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["metrics"]["items_seen"] == 4
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0  # never overwrite an existing report
    cmd[-1] = str(root / "forbidden.json")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and not (root / "forbidden.json").exists()


@pytest.mark.parametrize("entry", ["evaluate", "diagnose"])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_cpu_inference_cli_accepts_amp_checkpoint(tmp_path, entry, explicit_config):
    import hashlib
    import json
    import subprocess
    import sys
    from lunar_rm.config import PROJECT_ROOT
    from lunar_rm.data import dataset_from_config
    # Construct CPU weights with CUDA AMP training metadata; no GPU or optimizer required.
    cfg = resolve_config(experiment="N0", overrides={"data_root": str(build_dataset(tmp_path / "data")),
                         "device": "cuda", "amp": True, "width": 4, "groups": 2, "num_workers": 0})
    ds = dataset_from_config(cfg, "val")
    path = tmp_path / "amp.pt"
    torch.save({"format_version": 1, "model": build_model(cfg).state_dict(), "config": cfg.to_dict(),
                "data_identity": {"val": ds.identity}}, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cmd = [sys.executable, str(PROJECT_ROOT / f"scripts/{entry}.py"), "--checkpoint", str(path),
           "--device", "cpu", "--max-per-band", "1"]
    if explicit_config:
        config = tmp_path / "amp-config.json"
        config.write_text(json.dumps(cfg.to_dict()))
        cmd += ["--config", str(config)]
    if entry == "evaluate":
        output = tmp_path / "evaluation.json"
        cmd += ["--output", str(output)]
    else:
        manifest = tmp_path / "examples.json"
        manifest.write_text(json.dumps([{"band": "415", "sample_id": ds.rows[0]["sample_id"]}]))
        output = tmp_path / "diagnosis" / "diagnostics.json"
        cmd += ["--examples", str(manifest), "--output-dir", str(output.parent)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert report["config"]["device"] == "cpu" and report["config"]["amp"] is False
    assert report["target"] == "raw_db" and report["metrics"]["items_seen"] == 2
    assert report["checkpoint"]["sha256"] == before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["training_started"] is False
