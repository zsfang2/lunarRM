"""Read-only access to official consolidated array layouts."""

import csv
import hashlib
from pathlib import Path

import numpy as np


GRID = (256, 256)
BANDS = ("415", "58")


def split_paths(root, split):
    root = Path(root).resolve()
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")
    if (root / f"{split}_index.csv").is_file():
        directory, prefix = root, f"{split}_"
    elif (root / split / "index.csv").is_file():
        directory, prefix = root / split, ""
    else:
        raise FileNotFoundError(f"no official index for split {split!r} in {root}")
    return {name: directory / f"{prefix}{name}{'.csv' if name == 'index' else '.npy'}"
            for name in ("index", "hm", "pl415", "pl58", "mask415", "mask58")}


def read_index(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "terrain_id", "hm_row", "tx_row", "tx_col", "tx_index", "group"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"index missing required columns: {sorted(required - set(reader.fieldnames or ())) }")
        rows = list(reader)
    if not rows:
        raise ValueError("index is empty")
    return rows


def file_identity(path, *, content_hash=False):
    """Array identity uses header plus file version, not a full content checksum."""
    path = Path(path)
    st = path.stat()
    result = {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if content_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def open_array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def validate_index_rows(rows, terrain_count, errors, split):
    sample_ids, terrains, terrain_to_hm, hm_to_terrain = set(), set(), {}, {}
    for i, row in enumerate(rows):
        sid, tid = row["sample_id"], row["terrain_id"]
        if not sid or not tid:
            errors.append(f"{split} row {i}: empty sample/terrain identity")
        if sid in sample_ids:
            errors.append(f"{split} row {i}: duplicate sample_id {sid}")
        sample_ids.add(sid)
        terrains.add(tid)
        try:
            h, r, c = (int(row[k]) for k in ("hm_row", "tx_row", "tx_col"))
            if not 0 <= h < terrain_count:
                errors.append(f"{split} row {i}: hm_row out of bounds")
            if not (0 <= r < GRID[0] and 0 <= c < GRID[1]):
                errors.append(f"{split} row {i}: TX out of bounds")
            if terrain_to_hm.setdefault(tid, h) != h or hm_to_terrain.setdefault(h, tid) != tid:
                errors.append(f"{split} row {i}: inconsistent terrain/hm_row mapping")
        except (ValueError, TypeError):
            errors.append(f"{split} row {i}: invalid hm_row or TX integer")
    if set(hm_to_terrain) != set(range(terrain_count)):
        errors.append(f"{split}: hm_row coverage does not match heightmaps")
    return terrains
