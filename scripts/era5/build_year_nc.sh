#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=""
for candidate in "$PWD" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" "$SCRIPT_DIR/../.."; do
  [[ -n "$candidate" ]] || continue
  candidate="$(cd "$candidate" 2>/dev/null && pwd)" || continue
  if [[ -f "$candidate/setup.py" && -d "$candidate/scripts/era5" ]]; then
    REPO_ROOT="$candidate"
    break
  fi
done
[[ -n "$REPO_ROOT" ]] || {
  echo "Could not locate EcoPerceiver repo root." >&2
  exit 2
}
cd "$REPO_ROOT"

if [[ -n "${ECOPERCEIVER_ENV:-}" && -f "$ECOPERCEIVER_ENV/bin/activate" ]]; then
  # Optional override for standalone use outside the usual SCRATCH environment.
  source "$ECOPERCEIVER_ENV/bin/activate"
elif [[ -n "${SCRATCH:-}" && -f "$SCRATCH/env/ecoperceiver/bin/activate" ]]; then
  source "$SCRATCH/env/ecoperceiver/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 -u eval/era5/build_year_nc_from_shards.py "$@"
