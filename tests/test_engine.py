import json
from dataclasses import replace

import pytest
import torch

from lunar_rm.config import resolve_config
from lunar_rm.engine import train
from lunar_rm.checkpoint import read_checkpoint, atomic_save
from lunar_rm.evaluation import evaluate_checkpoint
from tests.fixtures.official import build_dataset


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(tmp_path):
    return resolve_config(experiment="N0", overrides={"data_root": str(build_dataset(tmp_path / "data")),
                          "output_dir": str(tmp_path / "continuous"), "device": "cpu", "num_workers": 0,
                          "epochs": 2, "batch_size": 6, "width": 2, "groups": 1})


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_epoch_boundary_resume_is_exact_and_independently_evaluable(tmp_path):
    cfg = config(tmp_path)
    train(cfg)
    split = replace(cfg, output_dir=str(tmp_path / "split"))
    train(split, until_epoch=1)
    split = replace(split, resume=str(tmp_path / "split/last.pt"))
    train(split)
    full = read_checkpoint(tmp_path / "continuous/last.pt")
    resumed = read_checkpoint(tmp_path / "split/last.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "sampler", "global_step", "best_metric", "validation"):
        assert_nested_equal(full[key], resumed[key])
    assert full["epoch"] == 2 and full["global_step"] == 2
    assert evaluate_checkpoint(replace(cfg, num_workers=0), tmp_path / "continuous/last.pt")["metrics"] == full["validation"]
    logs = [json.loads(s) for s in (tmp_path / "continuous/metrics.jsonl").read_text().splitlines()]
    assert logs[-1]["skipped_items"] == 2 and logs[-1]["updates"] == 1
    assert logs[-1]["valid_pixels"] > 0
    assert logs[0]["lr"] > logs[1]["lr"]
    full["optimizer"]["state"] = {}
    atomic_save(full, tmp_path / "reset.pt")
    with pytest.raises(ValueError, match="optimizer"):
        train(replace(cfg, resume=str(tmp_path / "reset.pt"), output_dir=str(tmp_path / "bad-output")))


def test_empty_training_mask_never_steps_and_incompatible_resume_rejected(tmp_path):
    import numpy as np
    cfg = config(tmp_path)
    for band in cfg.bands:
        p = tmp_path / f"data/train_mask{band}.npy"
        a = np.load(p); a.fill(0); np.save(p, a)
    train(cfg, until_epoch=1)
    state = read_checkpoint(tmp_path / "continuous/last.pt")
    assert state["global_step"] == 0 and not state["optimizer"]["state"]
    resumed = replace(cfg, resume=str(tmp_path / "continuous/last.pt"))
    with pytest.raises(ValueError, match="config"):
        train(replace(resumed, loss="mse"))
    state["scheduler"]["last_epoch"] = 0
    atomic_save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="scheduler"):
        train(replace(resumed, resume=str(tmp_path / "bad.pt"), output_dir=str(tmp_path / "new")))
    assert not (tmp_path / "new").exists()
    state["scheduler"]["last_epoch"] = 1
    state["data_identity"]["train"] = {}
    atomic_save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="data identity"):
        train(replace(resumed, resume=str(tmp_path / "bad.pt"), output_dir=str(tmp_path / "new")))


def test_nonfinite_prediction_rejected_before_checkpoint(tmp_path, monkeypatch):
    from lunar_rm.models import RadioMapModel
    cfg = config(tmp_path)
    original = RadioMapModel.forward
    def invalid(self, *args):
        return original(self, *args) * float("nan")
    monkeypatch.setattr(RadioMapModel, "forward", invalid)
    with pytest.raises(FloatingPointError):
        train(cfg)
    assert not (tmp_path / "continuous/last.pt").exists()


def test_atomic_save_preserves_old_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    atomic_save({"value": 1}, path)
    old = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"value": 2}, path)
    assert path.read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.pt"]


def test_dry_run_import_and_cuda_rejection_have_no_updates(tmp_path, monkeypatch):
    import subprocess
    import sys
    from lunar_rm.config import PROJECT_ROOT
    cfg = config(tmp_path)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/train.py"), "--experiment", "N0",
           "--device", "cpu", "--data-root", cfg.data_root, "--output-dir", cfg.output_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["training_started"] is False
    assert not (tmp_path / "continuous").exists()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        train(replace(cfg, device="cuda"))
    assert not (tmp_path / "continuous").exists()


def test_training_spawn_matches_single_process(tmp_path, monkeypatch):
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    cfg = config(tmp_path)
    train(cfg, until_epoch=1)
    train(replace(cfg, num_workers=1, output_dir=str(tmp_path / "workers")), until_epoch=1)
    a = read_checkpoint(tmp_path / "continuous/last.pt")
    b = read_checkpoint(tmp_path / "workers/last.pt")
    for key in ("model", "optimizer", "validation", "rng"):
        assert_nested_equal(a[key], b[key])
