"""Small consolidated arrays using the official 256x256 storage format."""

import csv
import json

import numpy as np


COUNTS = {"train": (3, 2), "val": (2, 1)}


def build_dataset(root, *, nested=False):
    root.mkdir(parents=True, exist_ok=True)
    meta = {"grid": [256, 256], "resolution_m_per_px": 1.0,
            "heightmap_range_m": [0.0, 496.0], "pathloss_range_db": [20.0, 228.0],
            "bands_hz": {"415": 415e6, "58": 5775e6},
            "splits": {s: {"samples_per_band": n, "terrains": t, "has_targets": True}
                       for s, (n, t) in COUNTS.items()}}
    (root / "metadata.json").write_text(json.dumps(meta))
    for split, (n, terrains) in COUNTS.items():
        d = root / split if nested else root
        d.mkdir(exist_ok=True)
        prefix = "" if nested else f"{split}_"
        rr, cc = np.indices((256, 256))
        hm = np.stack([rr * 0.5 + cc * 0.25 + i * 5 for i in range(terrains)]).astype("float32")
        np.save(d / f"{prefix}hm.npy", hm)
        rows = []
        for i in range(n):
            h = (i + 1) % terrains
            terrain = f"{h + (0 if split == 'train' else 10):05d}"
            rows.append({"sample_id": f"{terrain}_{i:02d}", "terrain_id": terrain,
                         "tx_index": str(i), "group": "multi", "tx_row": str(i * 7),
                         "tx_col": str(255 - i * 9), "hm_row": str(h)})
        write_index(d / f"{prefix}index.csv", rows)
        for band, offset in (("415", 0), ("58", 50)):
            pl = np.stack([np.full((256, 256), 80 + offset + i, dtype="float16") for i in range(n)])
            mask = np.ones(pl.shape, dtype="uint8")
            pl[0, 0, :3] = [145, 10, 240]
            mask[0, 1, 0] = 0
            mask[-1] = 0
            np.save(d / f"{prefix}pl{band}.npy", pl)
            np.save(d / f"{prefix}mask{band}.npy", np.packbits(mask.reshape(n, -1), axis=1))
    return root


def write_index(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
