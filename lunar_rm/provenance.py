"""Record local code identity without modifying the repository."""

import hashlib
import subprocess
from .config import PROJECT_ROOT


def code_identity():
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        run = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], text=True,
                             capture_output=True, timeout=10)
        result[key] = run.stdout.strip() if run.returncode == 0 else None
    paths = sorted([*PROJECT_ROOT.glob("lunar_rm/*.py"), *PROJECT_ROOT.glob("scripts/*.py"),
                    *PROJECT_ROOT.glob("configs/*.json"), PROJECT_ROOT / "model.py"])
    result["source_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths}
    return result
