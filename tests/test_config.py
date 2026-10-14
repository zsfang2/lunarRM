import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lunar_rm.config import PROJECT_ROOT, resolve_config, validate_runtime
from lunar_rm.registry import EXPERIMENTS


def test_all_shipped_configs_and_matched_controls():
    for path in sorted((PROJECT_ROOT / "configs").glob("*.json")):
        cfg = resolve_config(path)
        assert cfg.experiment in EXPERIMENTS
        assert cfg.to_dict()["features"] == list(cfg.features)
        assert cfg.bands == ("415", "58")
    assert {p.stem for p in (PROJECT_ROOT / "configs").glob("*.json")} == set(EXPERIMENTS)
    r, g = resolve_config(experiment="R0"), resolve_config(experiment="G0")
    assert r.features == g.features == ("height", "tx", "band", "fspl")
    assert (r.width, r.groups, r.epochs, r.loss) == (g.width, g.groups, g.epochs, g.loss)
    assert (r.head, g.head) == ("residual", "direct")


def test_override_precedence(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"experiment": "R0", "epochs": 7, "width": 24}))
    cfg = resolve_config(path, overrides={"epochs": 3})
    assert (cfg.epochs, cfg.width, cfg.loss) == (3, 24, "mae")
    assert resolve_config(experiment="N0").epochs == 20
    assert resolve_config(experiment="R0").width == 16
    with pytest.raises(ValueError, match="experiment"):
        resolve_config(path, experiment="N0")


@pytest.mark.parametrize("override", [
    {"epochs": -1}, {"epochs": True}, {"epochs": 0}, {"batch_size": 0},
    {"width": 10}, {"groups": 0}, {"width": 0}, {"num_workers": -1},
    {"lr": float("nan")}, {"weight_decay": -0.1}, {"nlos_alpha": -1},
    {"features": ["height", "tx", "tx"]}, {"features": ["target_db"]},
    {"features": ["valid_mask"]}, {"features": "height"},
    {"bands": ["415", "415"]}, {"bands": ["both"]}, {"bands": []},
    {"loss": "huber"}, {"device": "auto"}, {"device": "cuda:-1"},
    {"amp": "false"}, {"amp": True, "device": "cpu"},
    {"profile_step_m": 0}, {"receiver_chunk_size": 0}, {"unknown": 1},
    {"model": "unet"}, {"head": "direct"},
])
def test_invalid_config_rejected(override):
    with pytest.raises(ValueError):
        resolve_config(experiment="R0", overrides=override)


def test_unknown_experiment_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match="experiment"):
        resolve_config(experiment="fake")
    path = tmp_path / "bad.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        resolve_config(path)


def test_ablation_changes_only_intended_factor():
    reference = resolve_config(experiment="R2").to_dict()
    cases = {"R2_no_los": "features", "R2_no_fresnel": "features",
             "R2_no_curvature": "features", "R2_mse": "loss",
             "R2_alpha01": "nlos_alpha", "R2_alpha05": "nlos_alpha",
             "R2_direct": "head"}
    for name, factor in cases.items():
        other = resolve_config(experiment=name).to_dict()
        assert {key for key in reference if reference[key] != other[key]} == {
            "experiment", "output_dir", factor}
    for alias in ("R2_mae", "R2_alpha0"):
        other = resolve_config(experiment=alias).to_dict()
        assert {key for key in reference if reference[key] != other[key]} == {
            "experiment", "output_dir"}


def test_runtime_cuda_failure_and_index(monkeypatch, tmp_path):
    cfg = resolve_config(overrides={"device": "cuda", "output_dir": str(tmp_path / "out")})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        validate_runtime(cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    cfg = resolve_config(overrides={"device": "cuda:1"})
    with pytest.raises(ValueError, match="CUDA"):
        validate_runtime(cfg)


def test_output_collision_resume_and_data_protection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "run"
    cfg = resolve_config(overrides={"device": "cpu", "data_root": str(data), "output_dir": str(out)})
    validate_runtime(cfg, for_training=True)
    assert not out.exists()
    out.mkdir()
    with pytest.raises(ValueError, match="exists"):
        validate_runtime(cfg, for_training=True)
    checkpoint = out / "last.pt"
    checkpoint.write_bytes(b"content validation belongs to checkpoint loader")
    resumed = resolve_config(overrides={**cfg.to_dict(), "resume": str(checkpoint)})
    validate_runtime(resumed, for_training=True)
    with pytest.raises(ValueError, match="resume"):
        validate_runtime(resolve_config(overrides={**cfg.to_dict(), "resume": str(out / "missing.pt")}))
    for forbidden in (data, data / "runs", tmp_path):
        with pytest.raises(ValueError, match="data"):
            resolve_config(overrides={**cfg.to_dict(), "output_dir": str(forbidden)})
    link = tmp_path / "link"
    link.symlink_to(data, target_is_directory=True)
    with pytest.raises(ValueError, match="data"):
        resolve_config(overrides={**cfg.to_dict(), "output_dir": str(link / "run")})


def test_cli_help_and_dry_run_do_not_update_weights(monkeypatch, tmp_path, capsys):
    from lunar_rm.config import main

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration must not initialize training or save weights")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    before = torch.get_rng_state().clone()
    out = tmp_path / "run"
    assert main(["--experiment", "R0", "--device", "cpu", "--output-dir", str(out), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dry-run" and result["training_started"] is False
    assert result["config"]["head"] == "residual"
    assert torch.equal(before, torch.get_rng_state())
    assert not out.exists()
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("flag", ["-gpu", "--gpu"])
def test_training_gpu_alias_selects_visible_device(tmp_path, monkeypatch, capsys, flag):
    from scripts.train import main as train_main
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    output = tmp_path / "unused"
    train_main([flag, "1", "--output-dir", str(output), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["device"] == "cuda:1"
    assert report["training_started"] is False and not output.exists()
    for arguments in ([flag, "-1"], [flag, "2"], [flag, "0", "--device", "cpu"]):
        with pytest.raises(SystemExit) as error:
            train_main([*arguments, "--output-dir", str(output)])
        assert error.value.code == 2
    assert not output.exists()


def test_cli_invocation_outside_repo(tmp_path):
    script = PROJECT_ROOT / "scripts" / "check_config.py"
    result = subprocess.run([sys.executable, str(script), "--config", "configs/G0.json", "--check"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["config"]["experiment"] == "G0"
