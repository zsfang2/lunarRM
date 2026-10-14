import copy
import json
import re
import subprocess
import sys

import pytest

from lunar_rm.config import PROJECT_ROOT, resolve_config
from lunar_rm.registry import EXPERIMENTS
from lunar_rm.results import summarize, result_template


def evaluation():
    return {"config": resolve_config(experiment="A0").to_dict(),
            "data_identity": {"val": {"loader": "fixture"}},
            "code_identity": {"commit": "fixture"}, "checkpoint": None,
            "training_started": False, "target": "raw_db",
            "metrics": {"scope": "subset", "items_seen": 2, "items_expected": 2,
                        "full_dataset_items": 4, "max_per_band": 1,
                        "bands": {b: {"rmse_db": 3., "mae_db": 2., "valid_pixels": 10}
                                  for b in ("415", "58")}, "equal_band_rmse_db": 3.}}


def test_catalog_preserves_scope_provenance_and_flags_differences(tmp_path):
    first, second = evaluation(), evaluation()
    second["config"]["seed"] += 1
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    for path, report in zip(paths, (first, second)):
        path.write_text(json.dumps(report))
    result = summarize(paths)
    assert result["records"][0]["scope"] == "subset"
    assert result["records"][0]["report"] == first
    assert result["records"][0]["weights"] == "analytic"
    assert result["differences_from_first"][1]["config_fields"] == ["seed"]
    assert result["records"][0]["source_sha256"]
    assert not result["ranked"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize([paths[0], paths[0]])
    for edit in (lambda r: r.pop("data_identity"),
                 lambda r: r.update(target="clipped"),
                 lambda r: r["metrics"].update(scope="full")):
        bad = copy.deepcopy(first)
        edit(bad)
        paths[0].write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            summarize([paths[0]])


def test_template_is_pending_and_documented_paths_exist():
    template = result_template()
    assert json.loads((PROJECT_ROOT / "docs/results-template.json").read_text()) == template
    assert [r["experiment"] for r in template["records"]] == list(EXPERIMENTS)
    for row in template["records"]:
        assert row["status"] == "not_run"
        for field in ("metrics", "diagnostics", "benchmark", "checkpoint", "data_identity", "code_identity"):
            assert row[field] is None
    readme = (PROJECT_ROOT / "README.md").read_text()
    for name in ("audit_data", "check_forward", "check_config", "train", "evaluate", "diagnose", "benchmark", "summarize"):
        assert f"scripts/{name}.py" in readme
    for path in re.findall(r"scripts/[a-z_]+\.py", readme):
        assert (PROJECT_ROOT / path).is_file()
    assert "目前没有正式训练结果" in readme
    assert "训练引擎尚待实现" not in readme


def test_summary_cli_and_legacy_training_entry_are_safe(tmp_path):
    out = tmp_path / "results"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/summarize.py"), "--template", "--output-dir", str(out)]
    run = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "results.json").read_text())["records"][0]["status"] == "not_run"
    assert (out / "results.csv").is_file()
    assert subprocess.run(cmd, capture_output=True, timeout=30).returncode != 0
    for entry in ([sys.executable, "train.py"], ["bash", "run_train.sh"]):
        run = subprocess.run([*entry, "--device", "cpu", "--output-dir", str(tmp_path / "absent")],
                             cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert not json.loads(run.stdout)["training_started"]
    assert not (tmp_path / "absent").exists()


def test_benchmark_catalog_stays_separate_and_data_is_read_only(tmp_path):
    from lunar_rm.results import write_catalog
    report = evaluation()
    report.pop("metrics")
    report.pop("target")
    report.update(weights="untrained", timings={"end_to_end": {"median_ms": 12, "p95_ms": 15}},
                  hardware={"device": "cpu"}, boundaries={"scope": "single sample"},
                  parameters={"total": 1}, complexity={"complete": False})
    report["config"] = resolve_config(experiment="N0", overrides={"data_root": str(tmp_path / "data")}).to_dict()
    source = tmp_path / "timing.json"
    source.write_text(json.dumps(report))
    result = summarize([source])
    row = result["records"][0]
    assert row["scope"] == "single_sample" and row["weights"] == "untrained"
    assert "metrics" not in row["report"]
    with pytest.raises(ValueError, match="separate"):
        write_catalog(result, tmp_path / "data" / "report")
    assert not (tmp_path / "data").exists()
    report["weights"] = "checkpoint"
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checkpoint"):
        summarize([source])


def test_all_configs_dry_run(tmp_path):
    for name in EXPERIMENTS:
        run = subprocess.run([sys.executable, "scripts/check_config.py", "--config", f"configs/{name}.json",
                              "--device", "cpu", "--output-dir", str(tmp_path / name), "--dry-run"],
                             cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert not (tmp_path / name).exists()
