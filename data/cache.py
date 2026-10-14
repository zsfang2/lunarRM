"""Versioned atomic feature cache, separate from read-only source arrays."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class FeatureCache:
    def __init__(self, root, *, data_root):
        self.root = Path(root).resolve()
        data_root = Path(data_root).resolve()
        if self.root.is_relative_to(data_root) or data_root.is_relative_to(self.root):
            raise ValueError("cache must be separate from the read-only data root")

    def _path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid cache key")
        return self.root / f"{key}.npz"

    def get(self, key, identity, expected_fields, shape):
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved["metadata"].item())) != identity:
                    raise ValueError("cache identity/version mismatch")
                if set(saved.files) - {"metadata"} != set(expected_fields):
                    raise ValueError("cache fields incomplete or unexpected")
                result = {}
                for name in expected_fields:
                    array = saved[name]
                    dtype = np.dtype(bool if name == "los" else identity["dtype"])
                    if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                        raise ValueError(f"cache shape/dtype/finite check failed: {name}")
                    result[name] = array
                return result
        except (OSError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"invalid feature cache: {path}") from exc

    def put(self, key, identity, arrays):
        if cache_key(identity) != key:
            raise ValueError("cache key disagrees with identity")
        path = self._path(key)
        self.root.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".npz", delete=False) as handle:
                temp_path = Path(handle.name)
                np.savez_compressed(handle, metadata=json.dumps(identity, sort_keys=True), **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
