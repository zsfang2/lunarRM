#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# One-command training; explicit config/experiment/resume and dry-run take priority.
export MKL_THREADING_LAYER=GNU
defaults=()
has_profile=false
has_mode=false
for arg in "$@"; do
  case "$arg" in
    --config|--config=*|--experiment|--experiment=*|--resume|--resume=*) has_profile=true ;;
    --execute|--dry-run) has_mode=true ;;
  esac
done
if ! "$has_profile"; then defaults+=(--experiment R2); fi
if ! "$has_mode"; then defaults+=(--execute); fi

exec python scripts/train.py "${defaults[@]}" "$@"
