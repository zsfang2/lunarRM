"""Lossless report catalogs; do not rank or silently combine unlike experiments."""

import csv
import hashlib
import json
from pathlib import Path

from .registry import EXPERIMENTS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def result_template():
    return {"schema_version": 1, "ranked": False, "records": [
        {"experiment": name, "status": "not_run", "config_path": f"configs/{name}.json",
         "config": None, "checkpoint": None, "data_identity": None, "code_identity": None,
         "metrics": None, "diagnostics": None, "examples_manifest": None, "benchmark": None}
        for name in EXPERIMENTS],
        "notes": "Null means not measured. No formal training or test performance is claimed. "
                 "Keep raw-dB validation bands/counts and grouped diagnostics; benchmark must retain "
                 "feature/network/uncached end-to-end/cache lanes, parameters, partial MAC coverage, "
                 "memory, hardware, threads, precision, warmup, repeats and boundaries."}


def summarize(paths):
    """Keep one row per artifact and full source payloads, with config differences."""
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("supply nonempty, nonduplicate report paths")
    records = []
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        _digest(report)  # Also rejects NaN/Infinity in fields not flattened to CSV.
        if not isinstance(report, dict):
            raise ValueError(f"expected report object: {path}")
        for key in ("config", "data_identity", "code_identity"):
            if not isinstance(report.get(key), dict) or not report[key]:
                raise ValueError(f"missing {key}: {path}")
        cfg = report["config"]
        if cfg.get("experiment") not in EXPERIMENTS or not cfg.get("data_root"):
            raise ValueError(f"missing experiment/data_root: {path}")
        checkpoint = report.get("checkpoint")
        if checkpoint is not None and not checkpoint.get("sha256"):
            raise ValueError(f"checkpoint SHA256 required: {path}")
        if "metrics" in report:
            metric = report["metrics"]
            scope = metric.get("scope")
            if report.get("target") != "raw_db" or scope not in ("full", "subset"):
                raise ValueError(f"expected raw_db full/subset evaluation: {path}")
            if metric.get("items_seen") != metric.get("items_expected"):
                raise ValueError(f"incomplete evaluation coverage: {path}")
            if scope == "full" and (metric.get("max_per_band") is not None or
                                    metric.get("items_seen") != metric.get("full_dataset_items")):
                raise ValueError(f"subset cannot be relabelled full: {path}")
            if set(metric.get("bands", {})) != {"415", "58"}:
                raise ValueError(f"missing band metrics: {path}")
            kind, weights = "evaluation", "analytic" if cfg.get("model") == "fspl" else "checkpoint"
            if weights == "checkpoint" and checkpoint is None:
                raise ValueError(f"learned evaluation requires checkpoint identity: {path}")
        elif "timings" in report:
            kind, scope, weights = "benchmark", "single_sample", report.get("weights")
            if weights not in ("analytic", "checkpoint", "untrained"):
                raise ValueError(f"missing weight status: {path}")
            if (weights == "checkpoint") != (checkpoint is not None):
                raise ValueError(f"inconsistent benchmark checkpoint identity: {path}")
            for key in ("hardware", "boundaries", "parameters", "complexity"):
                if not report.get(key):
                    raise ValueError(f"missing benchmark {key}: {path}")
        else:
            raise ValueError(f"unsupported report; use evaluate/diagnose/benchmark JSON: {path}")
        records.append({"experiment": cfg["experiment"], "status": "reported", "kind": kind,
                        "scope": scope, "weights": weights, "source": str(path),
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "config_sha256": _digest(cfg), "data_sha256": _digest(report["data_identity"]),
                        "code_sha256": _digest(report["code_identity"]), "report": report})
    first = records[0]["report"]
    differences = []
    for record in records:
        report = record["report"]
        differences.append({"source": record["source"],
                            "config_fields": sorted(k for k in set(first["config"]) | set(report["config"])
                                                    if first["config"].get(k) != report["config"].get(k)),
                            "data_differs": first["data_identity"] != report["data_identity"],
                            "code_differs": first["code_identity"] != report["code_identity"],
                            "hardware_differs": first.get("hardware") != report.get("hardware")})
    return {"schema_version": 1, "ranked": False, "records": records,
            "differences_from_first": differences,
            "notes": "One row per source; no merging, ranking, seed aggregation or RT speedup. "
                     "Full/subset validation and single-sample timing retain distinct scopes. "
                     "Inspect complete config/data/code/weight/hardware provenance before comparison."}


def write_catalog(catalog, directory):
    directory = Path(directory).resolve()
    roots = [Path(r["report"]["config"]["data_root"]).resolve()
             for r in catalog["records"] if "report" in r]
    from .config import Config
    roots.append(Path(Config().data_root).resolve())
    if directory.exists() or any(directory.is_relative_to(root) or root.is_relative_to(directory) for root in roots):
        raise ValueError("output must be a new directory separate from data roots")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "results.json").write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    fields = ["experiment", "status", "kind", "scope", "weights", "source", "source_sha256",
              "config_sha256", "data_sha256", "code_sha256", "checkpoint_sha256",
              "items_seen", "rmse_415_db", "mae_415_db", "rmse_58_db", "mae_58_db",
              "equal_band_rmse_db", "feature_median_ms", "network_median_ms", "end_to_end_median_ms",
              "end_to_end_p95_ms", "cache_inputs_median_ms", "hardware", "boundaries"]
    with (directory / "results.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in catalog["records"]:
            row = {k: record.get(k) for k in fields}
            report = record.get("report", {})
            row["checkpoint_sha256"] = (report.get("checkpoint") or {}).get("sha256")
            metric = report.get("metrics", {})
            row.update(items_seen=metric.get("items_seen"), equal_band_rmse_db=metric.get("equal_band_rmse_db"))
            for band in ("415", "58"):
                for measure in ("rmse", "mae"):
                    row[f"{measure}_{band}_db"] = metric.get("bands", {}).get(band, {}).get(f"{measure}_db")
            for column, lane, stat in (("feature_median_ms", "feature_generation", "median_ms"),
                                       ("network_median_ms", "network_forward", "median_ms"),
                                       ("end_to_end_median_ms", "end_to_end", "median_ms"),
                                       ("end_to_end_p95_ms", "end_to_end", "p95_ms"),
                                       ("cache_inputs_median_ms", "cache_hit_inputs", "median_ms")):
                row[column] = (report.get("timings", {}).get(lane) or {}).get(stat)
            for key in ("hardware", "boundaries"):
                row[key] = json.dumps(report[key], sort_keys=True) if key in report else None
            writer.writerow(row)
