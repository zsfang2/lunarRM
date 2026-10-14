import pytest
import torch

from lunar_rm.benchmark import benchmark_model
from lunar_rm.config import resolve_config
from lunar_rm.data import dataset_from_config
from lunar_rm.models import build_model
from tests.fixtures.official import build_dataset


def test_benchmark_lanes_no_optimizer_or_cache_confusion(tmp_path):
    torch.set_num_threads(1)
    cfg = resolve_config(experiment="R0", overrides={"data_root": str(build_dataset(tmp_path / "data")),
                         "device": "cpu", "width": 4, "groups": 2, "cache_dir": str(tmp_path / "cache")})
    ds = dataset_from_config(cfg, "val")
    model = build_model(cfg)
    before = {k: v.clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
    report = benchmark_model(model, ds, 0, device="cpu", warmup=1, repeats=2)
    assert report["batch_size"] == 1
    assert report["uncached"] is True
    assert report["complexity"]["complete"] is False
    assert report["timings"]["cache_hit_inputs"]["median_ms"] >= 0
    assert report["timings"]["end_to_end"]["repeats"] == 2
    assert report["cpu_memory"]["process_lifetime_peak_rss_bytes"] > 0
    assert ds.cache is not None
    for k, v in before.items():
        assert torch.equal(v, model.state_dict()[k])
    assert all(p.grad is None for p in model.parameters())
    for args in ({"warmup": 0}, {"repeats": 0}):
        with pytest.raises(ValueError):
            benchmark_model(model, ds, 0, device="cpu", **args)


def test_label_free_input_builder(tmp_path):
    import numpy as np
    cfg = resolve_config(experiment="R0", overrides={"data_root": str(build_dataset(tmp_path / "data")), "device": "cpu"})
    ds = dataset_from_config(cfg, "val")
    item = ds[0]
    row = ds.rows[0]
    inputs = ds.build_inputs(item["raw_dem_m"][0].numpy(), (item["tx_row"], item["tx_col"]), row, "415")
    assert torch.equal(inputs["features"], item["features"])
    assert torch.equal(inputs["baseline_db"], item["baseline_db"])
    # Alter supervision on disk: the geometry-only call must be unchanged.
    target = np.load(ds.paths["pl415"], mmap_mode="r+")
    target[:] = 222
    target.flush()
    again = ds.build_inputs(item["raw_dem_m"][0].numpy(), (item["tx_row"], item["tx_col"]), row, "415")
    assert torch.equal(again["features"], inputs["features"])


def test_benchmark_cli_requires_weights_or_explicit_random(tmp_path):
    import json
    import subprocess
    import sys
    from lunar_rm.config import PROJECT_ROOT
    root = build_dataset(tmp_path / "data")
    output = tmp_path / "timing.json"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/benchmark.py"), "--experiment", "N0",
           "--data-root", str(root), "--device", "cpu", "--warmup", "1", "--repeats", "2",
           "--output", str(output)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert run.returncode != 0 and not output.exists()
    run = subprocess.run(cmd+["--allow-untrained"], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert result["weights"] == "untrained"
    assert result["checkpoint"] is None and not result["training_started"]
    assert result["timings"]["cache_hit_inputs"] is None
    assert result["complexity"]["conv_macs"] > 0
