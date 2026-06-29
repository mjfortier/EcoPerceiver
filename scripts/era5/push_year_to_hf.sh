#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
DEFAULT_HF_REPO_ID="ludolara/era5"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

usage() {
  cat >&2 <<EOF
Usage: $0 YEAR [options]

Assemble a calendar-year ERA5 NetCDF file, then upload it to a Hugging Face dataset.
The first step is always scripts/era5/build_year_nc.sh.

Options:
  --year YEAR                 Year to assemble and upload. Positional YEAR is also accepted.
  --run-path PATH             EcoPerceiver run directory.
  --input-dir PATH            Directory containing quarterly .nc files.
  --output-path PATH          Yearly NetCDF output path.
  --engine ENGINE             xarray NetCDF engine for assembly. Default: h5netcdf.
  --inputs Q1 Q2 Q3 Q4        Explicit quarter NetCDF paths for the assembler.
  --allow-missing             Pass through to build_year_nc.sh.
  --overwrite                 Replace an existing assembled output file.
  --hf-repo-id REPO_ID        Hugging Face dataset repo. Default: HF_REPO_ID or ludolara/era5.
  --path-in-repo PATH         Destination path in the dataset. Default: basename of output path.
  --revision REVISION         Branch/revision to upload to. Default: main.
  --commit-message MESSAGE    Commit message for the upload.
  --private                   Create the dataset as private if it does not already exist.
  --skip-create-repo          Do not create the dataset if it is missing.
  --dry-run                   Print assembler paths and upload target without writing.
  -h, --help                  Show this help.

Environment:
  HF_WRITE                    Hugging Face write token. Loaded from .env when present.
  HF_REPO_ID                  Optional default dataset repo id.
  PYTHON_BIN                  Optional Python executable for the Hugging Face upload.

Examples:
  $0 2017
  $0 --year 2017 --overwrite
  $0 2017 --path-in-repo predictions/era5_predictions_2017.nc
EOF
}

die() {
  echo "$1" >&2
  usage
  exit 2
}

YEAR="${YEAR:-}"
RUN_PATH="${RUN_PATH:-$DEFAULT_RUN_PATH}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
ENGINE="${XARRAY_ENGINE:-h5netcdf}"
ALLOW_MISSING=0
OVERWRITE=0
DRY_RUN=0
HF_REPO_ID="${HF_REPO_ID:-$DEFAULT_HF_REPO_ID}"
PATH_IN_REPO="${PATH_IN_REPO:-}"
REVISION="${HF_REVISION:-main}"
COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-}"
CREATE_REPO=1
PRIVATE=0
INPUTS=()

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${ECOPERCEIVER_ENV:-}" && -x "$ECOPERCEIVER_ENV/bin/python" ]]; then
    PYTHON_BIN="$ECOPERCEIVER_ENV/bin/python"
  elif [[ -n "${SCRATCH:-}" && -x "$SCRATCH/env/ecoperceiver/bin/python" ]]; then
    PYTHON_BIN="$SCRATCH/env/ecoperceiver/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --year)
      [[ $# -ge 2 ]] || die "--year requires a value."
      YEAR="$2"
      shift 2
      ;;
    --year=*)
      YEAR="${1#*=}"
      shift
      ;;
    --run-path)
      [[ $# -ge 2 ]] || die "--run-path requires a value."
      RUN_PATH="$2"
      shift 2
      ;;
    --run-path=*)
      RUN_PATH="${1#*=}"
      shift
      ;;
    --input-dir)
      [[ $# -ge 2 ]] || die "--input-dir requires a value."
      INPUT_DIR="$2"
      shift 2
      ;;
    --input-dir=*)
      INPUT_DIR="${1#*=}"
      shift
      ;;
    --output-path)
      [[ $# -ge 2 ]] || die "--output-path requires a value."
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --output-path=*)
      OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --engine)
      [[ $# -ge 2 ]] || die "--engine requires a value."
      ENGINE="$2"
      shift 2
      ;;
    --engine=*)
      ENGINE="${1#*=}"
      shift
      ;;
    --inputs)
      [[ $# -ge 5 ]] || die "--inputs requires four paths."
      INPUTS=("$2" "$3" "$4" "$5")
      shift 5
      ;;
    --allow-missing)
      ALLOW_MISSING=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --hf-repo-id)
      [[ $# -ge 2 ]] || die "--hf-repo-id requires a value."
      HF_REPO_ID="$2"
      shift 2
      ;;
    --hf-repo-id=*)
      HF_REPO_ID="${1#*=}"
      shift
      ;;
    --path-in-repo)
      [[ $# -ge 2 ]] || die "--path-in-repo requires a value."
      PATH_IN_REPO="$2"
      shift 2
      ;;
    --path-in-repo=*)
      PATH_IN_REPO="${1#*=}"
      shift
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "--revision requires a value."
      REVISION="$2"
      shift 2
      ;;
    --revision=*)
      REVISION="${1#*=}"
      shift
      ;;
    --commit-message)
      [[ $# -ge 2 ]] || die "--commit-message requires a value."
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --commit-message=*)
      COMMIT_MESSAGE="${1#*=}"
      shift
      ;;
    --private)
      PRIVATE=1
      shift
      ;;
    --skip-create-repo)
      CREATE_REPO=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "Unknown argument: $1"
      ;;
    *)
      if [[ -z "$YEAR" ]]; then
        YEAR="$1"
        shift
      else
        die "Unknown argument: $1"
      fi
      ;;
  esac
done

[[ -n "$YEAR" ]] || die "YEAR is required."
[[ "$YEAR" =~ ^[0-9]{4}$ ]] || die "YEAR must be a four-digit year, got: $YEAR"
[[ -n "$HF_REPO_ID" ]] || die "HF repo id is required."

if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$RUN_PATH/eval"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$INPUT_DIR/era5_predictions_${YEAR}.nc"
fi

if [[ -z "$PATH_IN_REPO" ]]; then
  PATH_IN_REPO="$(basename "$OUTPUT_PATH")"
fi
PATH_IN_REPO="${PATH_IN_REPO#/}"

if [[ -z "$COMMIT_MESSAGE" ]]; then
  COMMIT_MESSAGE="Upload ERA5 predictions for ${YEAR}"
fi

ASSEMBLE_ARGS=(
  --year "$YEAR"
  --run-path "$RUN_PATH"
  --input-dir "$INPUT_DIR"
  --output-path "$OUTPUT_PATH"
  --engine "$ENGINE"
)

if [[ "${#INPUTS[@]}" -gt 0 ]]; then
  ASSEMBLE_ARGS+=(--inputs "${INPUTS[@]}")
fi
if [[ "$ALLOW_MISSING" == "1" ]]; then
  ASSEMBLE_ARGS+=(--allow-missing)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ASSEMBLE_ARGS+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ASSEMBLE_ARGS+=(--dry-run)
fi

echo "[$(date)] Assembling yearly ERA5 NetCDF."
echo "Year: $YEAR"
echo "Output path: $OUTPUT_PATH"
"$SCRIPT_DIR/build_year_nc.sh" "${ASSEMBLE_ARGS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(date)] Dry-run upload target: dataset/$HF_REPO_ID@$REVISION:$PATH_IN_REPO"
  exit 0
fi

if [[ ! -f "$OUTPUT_PATH" ]]; then
  echo "Assembled output not found: $OUTPUT_PATH" >&2
  exit 1
fi

if [[ -z "${HF_WRITE:-}" ]]; then
  echo "HF_WRITE is required in the environment or .env." >&2
  exit 1
fi

echo "[$(date)] Uploading yearly ERA5 NetCDF to Hugging Face dataset."
echo "Destination: dataset/$HF_REPO_ID@$REVISION:$PATH_IN_REPO"

"$PYTHON_BIN" - "$HF_REPO_ID" "$OUTPUT_PATH" "$PATH_IN_REPO" "$REVISION" "$COMMIT_MESSAGE" "$CREATE_REPO" "$PRIVATE" <<'PY'
import os
from pathlib import Path
import sys

repo_id, output_path, path_in_repo, revision, commit_message, create_repo, private = sys.argv[1:8]
token = os.environ.get("HF_WRITE")

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    raise SystemExit(
        "Missing Python dependency while importing huggingface_hub: "
        f"{missing}. Install with: python3 -m pip install --user huggingface_hub filelock"
    ) from exc

api = HfApi()
if create_repo == "1":
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=(private == "1"),
        exist_ok=True,
    )

result = api.upload_file(
    path_or_fileobj=str(Path(output_path)),
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    token=token,
    commit_message=commit_message,
)
print(f"Upload complete: {result}")
PY
