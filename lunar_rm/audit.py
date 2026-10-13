"""Audit official arrays and fixed splits without modifying source data."""

import argparse
import json
from pathlib import Path

import numpy as np

from .storage import BANDS, GRID, file_identity, open_array, read_index, split_paths, validate_index_rows


OFFICIAL_COUNTS = {"train": (25720, 6120), "val": (2330, 680)}


def _finite_scan(array, limit, chunk_rows):
    count = 0
    for start in range(0, limit, chunk_rows):
        count += int(np.count_nonzero(~np.isfinite(array[start:min(start + chunk_rows, limit)])))
    return {"scanned_rows": limit, "total_rows": len(array), "nonfinite_values": count}


def _band_scan(pl, mask, limit, chunk_rows, bounds):
    result = {"scanned_rows": limit, "total_rows": len(pl), "valid_pixels": 0,
              "total_pixels": limit * GRID[0] * GRID[1], "nonfinite_values": 0,
              "valid_below_range": 0, "valid_above_range": 0, "empty_maps": 0,
              "coverage_min": None, "coverage_max": None}
    for start in range(0, limit, chunk_rows):
        stop = min(start + chunk_rows, limit)
        block = pl[start:stop]
        valid = np.unpackbits(mask[start:stop], axis=1, bitorder="big").reshape(block.shape).astype(bool)
        counts = valid.sum(axis=(1, 2), dtype=np.int64)
        result["valid_pixels"] += int(counts.sum())
        result["empty_maps"] += int(np.count_nonzero(counts == 0))
        result["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(block)))
        result["valid_below_range"] += int(np.count_nonzero(valid & (block < bounds[0])))
        result["valid_above_range"] += int(np.count_nonzero(valid & (block > bounds[1])))
        lo, hi = float(counts.min() / 65536), float(counts.max() / 65536)
        result["coverage_min"] = lo if result["coverage_min"] is None else min(lo, result["coverage_min"])
        result["coverage_max"] = hi if result["coverage_max"] is None else max(hi, result["coverage_max"])
    result["coverage_mean"] = result["valid_pixels"] / result["total_pixels"] if limit else None
    return result


def audit_dataset(root, *, max_rows=None, chunk_rows=32, expected_counts=None):
    """Scan all rows by default; max_rows limits *each array* to a prefix.

    expected_counts overrides are for synthetic tests, explicitly labeled in the
    report. Production CLI always uses the official fixed split counts.
    """
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer or None")
    root = Path(root).resolve()
    expected = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    report = {"root": str(root), "status": "failed", "full_scan": True, "errors": [],
              "scope": {"max_rows_per_array": max_rows, "chunk_rows": chunk_rows,
                        "row_selection": "all" if max_rows is None else "prefix",
                        "count_protocol": "official" if expected_counts is None else "custom_fixture"},
              "files": {}, "splits": {}, "identity_policy": "JSON/CSV/source SHA256; arrays shape/dtype/size/mtime, not full content hashes"}
    errors = report["errors"]
    try:
        meta = json.loads((root / "metadata.json").read_text())
        if meta["grid"] != list(GRID):
            raise ValueError("metadata grid must be 256x256")
        if meta["heightmap_range_m"] != [0, 496] or meta["pathloss_range_db"] != [20, 228]:
            raise ValueError("metadata normalization bounds do not match official public ranges")
        if meta["resolution_m_per_px"] != 1 or meta["bands_hz"] != {"415": 415e6, "58": 5775e6}:
            raise ValueError("metadata physical units/frequencies differ from official data")
        for name in ("metadata.json", "manifest.json", "lunar_dataset.py"):
            if (root / name).is_file():
                report["files"][name] = file_identity(root / name, content_hash=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"metadata: {exc}")
        report["full_scan"] = False
        return report
    terrain_sets = {}
    for split in ("train", "val"):
        try:
            paths = split_paths(root, split)
            rows = read_index(paths["index"])
            index_key = str(paths["index"].relative_to(root))
            report["files"][index_key] = file_identity(paths["index"], content_hash=True)
        except (OSError, ValueError) as exc:
            errors.append(f"{split}: {exc}")
            report["full_scan"] = False
            continue
        n = len(rows)
        terrains = len({row["terrain_id"] for row in rows})
        section = report["splits"][split] = {"rows": n, "terrains": terrains, "items_both": n * 2,
                                               "arrays": {}, "bands": {}}
        want_rows, want_terrains = expected[split]
        if n != want_rows:
            errors.append(f"{split}: expected {want_rows} index rows, got {n}")
        if terrains != want_terrains:
            errors.append(f"{split}: expected {want_terrains} terrains, got {terrains}")
        declared = meta.get("splits", {}).get(split, {})
        if declared.get("samples_per_band") != n or declared.get("terrains") != terrains:
            errors.append(f"{split}: metadata split counts disagree with index")
        arrays = {}
        for field in ("hm", "pl415", "pl58", "mask415", "mask58"):
            try:
                a = open_array(paths[field])
                shape = (terrains, *GRID) if field == "hm" else ((n, 8192) if field.startswith("mask") else (n, *GRID))
                dtype = "float32" if field == "hm" else ("uint8" if field.startswith("mask") else "float16")
                info = {**file_identity(paths[field]), "shape": list(a.shape), "dtype": str(a.dtype)}
                section["arrays"][field] = info
                report["files"][str(paths[field].relative_to(root))] = info
                if a.shape != shape:
                    raise ValueError(f"{field} shape {a.shape} != {shape}")
                if a.dtype != np.dtype(dtype):
                    raise ValueError(f"{field} dtype {a.dtype} != {dtype}")
                arrays[field] = a
            except (OSError, ValueError, EOFError) as exc:
                errors.append(f"{split}: {exc}")
                report["full_scan"] = False
        terrain_sets[split] = validate_index_rows(rows, len(arrays["hm"]) if "hm" in arrays else terrains, errors, split)
        if "hm" in arrays:
            a = arrays["hm"]
            limit = len(a) if max_rows is None else min(max_rows, len(a))
            section["heightmap"] = _finite_scan(a, limit, chunk_rows)
            report["full_scan"] &= limit == len(a)
            if section["heightmap"]["nonfinite_values"]:
                errors.append(f"{split}: non-finite DEM values")
        for band in BANDS:
            if f"pl{band}" not in arrays or f"mask{band}" not in arrays:
                continue
            limit = n if max_rows is None else min(max_rows, n)
            stats = _band_scan(arrays[f"pl{band}"], arrays[f"mask{band}"], limit, chunk_rows, meta["pathloss_range_db"])
            section["bands"][band] = stats
            report["full_scan"] &= limit == n
            if stats["nonfinite_values"]:
                errors.append(f"{split}/{band}: non-finite label values")
    if len(terrain_sets) == 2:
        overlap = terrain_sets["train"] & terrain_sets["val"]
        report["terrain_overlap_count"] = len(overlap)
        if overlap:
            errors.append(f"train/val terrain overlap: {sorted(overlap)[:10]}")
    report["status"] = "failed" if errors else ("passed" if report["full_scan"] else "partial_pass")
    return report


def save_report(report, output):
    output = Path(output).resolve()
    if output.is_relative_to(Path(report["root"]).resolve()):
        raise ValueError("audit output must be outside the read-only data root")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only audit of official LunarRM train/val arrays")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, help="scan only this prefix of each array; report remains partial")
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError("audit output must be outside the read-only data root")
        report = audit_dataset(args.data_root, max_rows=args.max_rows, chunk_rows=args.chunk_rows)
        if args.output:
            save_report(report, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
