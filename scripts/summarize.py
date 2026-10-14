"""Create pending result tables or catalog existing evaluation/benchmark JSON reports."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_rm.results import result_template, summarize, write_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.template == bool(args.reports):
        parser.error("choose --template or one or more report paths")
    try:
        catalog = result_template() if args.template else summarize(args.reports)
        write_catalog(catalog, args.output_dir)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Wrote results.json and results.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
