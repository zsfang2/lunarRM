"""Repository-local config checker; never executes training."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_rm.config import main


if __name__ == "__main__":
    raise SystemExit(main())
