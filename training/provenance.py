"""Record local code identity without modifying the repository."""

import hashlib
import subprocess
from training.config import PROJECT_ROOT


def code_identity():
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        run = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], text=True,
                             capture_output=True, timeout=10)
        result[key] = run.stdout.strip() if run.returncode == 0 else None
    paths = sorted([*(p for folder in ("data", "models", "training")
                      for p in (PROJECT_ROOT / folder).glob("*.py")),
                    *PROJECT_ROOT.glob("configs/*.json"), PROJECT_ROOT / "train.py",
                    PROJECT_ROOT / "evaluate.py", PROJECT_ROOT / "run_train.sh"])
    result["source_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths}
    return result
