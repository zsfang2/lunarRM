import csv
import hashlib
import json

import numpy as np
import pytest

from lunar_rm.audit import audit_dataset, save_report
from tests.fixtures.official import COUNTS, build_dataset, write_index


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("nested", [False, True])
def test_full_audit_is_read_only_and_counts_valid_outliers(tmp_path, nested):
    root = build_dataset(tmp_path / "data", nested=nested)
    before = snapshot(root)
    report = audit_dataset(root, expected_counts=COUNTS, chunk_rows=1)
    assert report["status"] == "passed" and report["full_scan"]
    assert not report["errors"]
    assert report["splits"]["train"]["items_both"] == 6
    stats = report["splits"]["train"]["bands"]["415"]
    assert stats["valid_pixels"] == 2 * 256 * 256 - 1
    assert stats["valid_below_range"] == stats["valid_above_range"] == 1
    assert stats["empty_maps"] == 1
    assert report["splits"]["train"]["arrays"]["hm"]["dtype"] == "float32"
    assert report["files"]["metadata.json"]["sha256"] == before["metadata.json"]
    assert snapshot(root) == before


def test_partial_audit_cannot_claim_full_pass(tmp_path):
    root = build_dataset(tmp_path / "data")
    report = audit_dataset(root, expected_counts=COUNTS, max_rows=1)
    assert report["status"] == "partial_pass" and not report["full_scan"]
    assert report["splits"]["train"]["bands"]["58"]["scanned_rows"] == 1
    assert report["splits"]["train"]["heightmap"]["scanned_rows"] == 1


@pytest.mark.parametrize("defect, expected", [
    ("hm_row", "hm_row"), ("tx_row", "TX"), ("duplicate", "duplicate sample_id"),
    ("overlap", "terrain overlap"), ("mask_length", "mask415 shape"),
    ("label_rows", "pl58 shape"), ("dtype", "hm dtype"),
    ("nonfinite", "non-finite"), ("mapping", "terrain/hm_row"),
])
def test_reject_corrupt_official_layout(tmp_path, defect, expected):
    root = build_dataset(tmp_path / "data")
    path = root / "train_index.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if defect in ("hm_row", "tx_row"):
        rows[0][defect] = "999"
    elif defect == "duplicate":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif defect == "overlap":
        rows[0]["terrain_id"] = "00010"
    elif defect == "mapping":
        rows[2]["hm_row"] = "0"
    write_index(path, rows)
    if defect == "mask_length":
        np.save(root / "train_mask415.npy", np.zeros((3, 8191), dtype="uint8"))
    elif defect == "label_rows":
        np.save(root / "train_pl58.npy", np.zeros((2, 256, 256), dtype="float16"))
    elif defect == "dtype":
        np.save(root / "train_hm.npy", np.zeros((2, 256, 256), dtype="float64"))
    elif defect == "nonfinite":
        a = np.load(root / "train_pl415.npy")
        a[0, 2, 2] = np.nan
        np.save(root / "train_pl415.npy", a)
    report = audit_dataset(root, expected_counts=COUNTS)
    assert report["status"] == "failed"
    assert any(expected in error for error in report["errors"]), report["errors"]


def test_official_counts_are_not_inferred_from_fixture_metadata(tmp_path):
    report = audit_dataset(build_dataset(tmp_path / "data"))
    assert any("expected 25720" in e for e in report["errors"])


def test_report_writes_cannot_modify_data(tmp_path):
    root = build_dataset(tmp_path / "data")
    report = audit_dataset(root, expected_counts=COUNTS)
    before = snapshot(root)
    with pytest.raises(ValueError, match="data root"):
        save_report(report, root / "audit.json")
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="data root"):
        save_report(report, link / "audit.json")
    target = tmp_path / "audit.json"
    save_report(report, target)
    assert json.loads(target.read_text())["status"] == "passed"
    with pytest.raises(FileExistsError):
        save_report(report, target)
    assert snapshot(root) == before


@pytest.mark.parametrize("options", [{"max_rows": 0}, {"max_rows": -1}, {"chunk_rows": 0}])
def test_invalid_scan_budget(tmp_path, options):
    with pytest.raises(ValueError):
        audit_dataset(build_dataset(tmp_path / "data"), expected_counts=COUNTS, **options)
