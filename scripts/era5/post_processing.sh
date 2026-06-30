#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/post_processing.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/post_processing.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=post-era5
#SBATCH --account=aip-pal
#SBATCH --partition=cpubase_bycore_b3

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

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ -n "${ECOPERCEIVER_ENV:-}" && -f "$ECOPERCEIVER_ENV/bin/activate" ]]; then
  source "$ECOPERCEIVER_ENV/bin/activate"
elif [[ -n "${SCRATCH:-}" && -f "$SCRATCH/env/ecoperceiver/bin/activate" ]]; then
  source "$SCRATCH/env/ecoperceiver/bin/activate"
fi

export PYTHONUNBUFFERED=1

DEFAULT_RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"

usage() {
  cat >&2 <<EOF
Usage: $0 YEAR [options]

Validate that all quarterly ERA5 shard CSVs exist, build the yearly NetCDF, then upload it.
The upload target defaults to Google Drive via scripts/era5/push_year.sh.

Options:
  --year YEAR                 Year to post-process. Positional YEAR is also accepted.
  --run-path PATH             EcoPerceiver run directory.
  --input-dir PATH            Directory containing quarterly shard directories.
  --output-path PATH          Yearly NetCDF output path.
  --shard-dirs Q1 Q2 Q3 Q4    Explicit quarter shard directories to validate and build from.
  --num-shards N              Expected rank CSV shards per quarter. Default: NUM_SHARDS or 4.
  --prediction-targets LIST   Prediction targets, comma or space separated.
  --netcdf-duplicate-policy P Duplicate policy: error, first, or last. Default: last.
  --chunk-rows N              CSV rows per builder chunk.
  --write-time-chunk N        Time steps per NetCDF write slice.
  --max-memory-gb GB          Builder allocation guard. Default: 470.
  --overwrite                 Replace an existing yearly NetCDF output file.
  --to TARGET                 Upload target: drive or hf. Default: PUSH_TARGET or drive.
  --target TARGET             Alias for --to.
  --push-target TARGET        Alias for --to.
  --path PATH                 Destination path passed to push_year.sh.
  --dry-run                   Print validation inputs and commands without writing.
  -h, --help                  Show this help.

Google Drive/rclone options:
  --rclone-remote REMOTE      rclone remote passed to push_year.sh.
  --drive-dir PATH            Google Drive folder/path passed to push_year.sh.
  --remote-dir REMOTE_DIR     Full rclone destination directory.
  --remote-path REMOTE_PATH   Full rclone destination file path.
  --rclone-arg ARG            Extra argument passed to rclone copyto. Repeatable.

Hugging Face options, only used with --to hf:
  --hf-repo-id REPO_ID        Dataset repo id passed to push_year.sh.
  --path-in-repo PATH         Destination path in the dataset.
  --revision REVISION         Branch/revision to upload to.
  --commit-message MESSAGE    Commit message for the upload.
  --private                   Create the dataset as private if it does not already exist.
  --skip-create-repo          Do not create the dataset if it is missing.

Environment:
  POST_PROCESS_PREDICTION_TARGETS
                              Optional default prediction targets for yearly NetCDF output.
  PREDICTION_TARGETS          Fallback default prediction targets.
  SHARD_READY_RETRIES         Validation attempts before failing. Default: 10.
  SHARD_READY_SLEEP_SECONDS   Seconds between validation attempts. Default: 60.
  PUSH_TARGET                 Optional default upload target. Default: drive.
  RCLONE_REMOTE               Optional default rclone remote, used by push_year.sh.
  DRIVE_DIR                   Optional default Drive folder/path, used by push_year.sh.
EOF
}

die() {
  echo "$1" >&2
  usage
  exit 2
}

append_prediction_targets() {
  local raw value
  raw="${1//,/ }"
  for value in $raw; do
    if [[ -n "$value" ]]; then
      PREDICTION_TARGET_LIST+=("$value")
    fi
  done
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

default_shard_dirs() {
  local year="$1"
  local input_dir="$2"
  SHARD_DIRS=(
    "$input_dir/.era5_predictions_${year}0101_to_${year}0331_multi_gpu_shards"
    "$input_dir/.era5_predictions_${year}0401_to_${year}0630_multi_gpu_shards"
    "$input_dir/.era5_predictions_${year}0701_to_${year}0930_multi_gpu_shards"
    "$input_dir/.era5_predictions_${year}1001_to_${year}1231_multi_gpu_shards"
  )
}

validate_shard_inputs() {
  local errors=0
  local expected_header=""
  local dir file header rank rank_name actual_count
  local files=()

  shopt -s nullglob
  for dir in "${SHARD_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
      echo "Missing shard directory: $dir" >&2
      errors=1
      continue
    fi

    files=("$dir"/rank_*.csv)
    actual_count="${#files[@]}"
    if [[ "$actual_count" -ne "$NUM_SHARDS" ]]; then
      echo "Expected $NUM_SHARDS rank_*.csv files in $dir, found $actual_count." >&2
      errors=1
    fi

    for ((rank = 0; rank < NUM_SHARDS; rank++)); do
      printf -v rank_name "rank_%05d.csv" "$rank"
      file="$dir/$rank_name"
      if [[ ! -f "$file" ]]; then
        echo "Missing shard file: $file" >&2
        errors=1
        continue
      fi
      if ! IFS= read -r header < "$file"; then
        echo "Shard file is empty: $file" >&2
        errors=1
        continue
      fi
      header="${header%$'\r'}"
      if [[ -z "$header" ]]; then
        echo "Shard file has an empty header: $file" >&2
        errors=1
        continue
      fi
      if [[ -z "$expected_header" ]]; then
        expected_header="$header"
      elif [[ "$header" != "$expected_header" ]]; then
        echo "CSV header mismatch in shard file: $file" >&2
        errors=1
      fi
    done
  done
  shopt -u nullglob

  return "$errors"
}

wait_for_shards() {
  local attempt=1
  while ! validate_shard_inputs; do
    if (( attempt >= SHARD_READY_RETRIES )); then
      echo "Shard validation failed after $attempt attempt(s); refusing to build yearly NetCDF." >&2
      return 1
    fi
    echo "Shard inputs are not complete yet; retrying in ${SHARD_READY_SLEEP_SECONDS}s ($attempt/$SHARD_READY_RETRIES)."
    sleep "$SHARD_READY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

YEAR="${YEAR:-}"
RUN_PATH="${RUN_PATH:-$DEFAULT_RUN_PATH}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"
POST_PROCESS_PREDICTION_TARGETS_ENV="${POST_PROCESS_PREDICTION_TARGETS:-${PREDICTION_TARGETS:-pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE}}"
PREDICTION_TARGET_LIST=()
append_prediction_targets "$POST_PROCESS_PREDICTION_TARGETS_ENV"
NETCDF_DUPLICATE_POLICY="${NETCDF_DUPLICATE_POLICY:-last}"
CHUNK_ROWS="${BUILD_YEAR_CHUNK_ROWS:-}"
WRITE_TIME_CHUNK="${BUILD_YEAR_WRITE_TIME_CHUNK:-}"
MAX_MEMORY_GB="${BUILD_YEAR_MAX_MEMORY_GB:-470}"
OVERWRITE=0
DRY_RUN=0
PUSH_TARGET="${PUSH_TARGET:-drive}"
DESTINATION_PATH="${DESTINATION_PATH:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
PATH_IN_REPO="${PATH_IN_REPO:-}"
REVISION="${REVISION:-${HF_REVISION:-}}"
COMMIT_MESSAGE="${HF_COMMIT_MESSAGE:-}"
CREATE_REPO="${CREATE_REPO:-1}"
PRIVATE="${PRIVATE:-0}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
DRIVE_DIR="${DRIVE_DIR:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
REMOTE_PATH="${REMOTE_PATH:-}"
RCLONE_ARGS=()
SHARD_READY_RETRIES="${SHARD_READY_RETRIES:-10}"
SHARD_READY_SLEEP_SECONDS="${SHARD_READY_SLEEP_SECONDS:-60}"
SHARD_DIRS=()

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
    --shard-dirs)
      [[ $# -ge 5 ]] || die "--shard-dirs requires four paths."
      SHARD_DIRS=("$2" "$3" "$4" "$5")
      shift 5
      ;;
    --num-shards)
      [[ $# -ge 2 ]] || die "--num-shards requires a value."
      NUM_SHARDS="$2"
      shift 2
      ;;
    --num-shards=*)
      NUM_SHARDS="${1#*=}"
      shift
      ;;
    --prediction-targets)
      shift
      PREDICTION_TARGET_LIST=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        append_prediction_targets "$1"
        shift
      done
      [[ "${#PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "--prediction-targets requires at least one target."
      ;;
    --prediction-targets=*)
      PREDICTION_TARGET_LIST=()
      append_prediction_targets "${1#*=}"
      [[ "${#PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "--prediction-targets requires at least one target."
      shift
      ;;
    --netcdf-duplicate-policy)
      [[ $# -ge 2 ]] || die "--netcdf-duplicate-policy requires a value."
      NETCDF_DUPLICATE_POLICY="$2"
      shift 2
      ;;
    --netcdf-duplicate-policy=*)
      NETCDF_DUPLICATE_POLICY="${1#*=}"
      shift
      ;;
    --chunk-rows)
      [[ $# -ge 2 ]] || die "--chunk-rows requires a value."
      CHUNK_ROWS="$2"
      shift 2
      ;;
    --chunk-rows=*)
      CHUNK_ROWS="${1#*=}"
      shift
      ;;
    --write-time-chunk)
      [[ $# -ge 2 ]] || die "--write-time-chunk requires a value."
      WRITE_TIME_CHUNK="$2"
      shift 2
      ;;
    --write-time-chunk=*)
      WRITE_TIME_CHUNK="${1#*=}"
      shift
      ;;
    --max-memory-gb)
      [[ $# -ge 2 ]] || die "--max-memory-gb requires a value."
      MAX_MEMORY_GB="$2"
      shift 2
      ;;
    --max-memory-gb=*)
      MAX_MEMORY_GB="${1#*=}"
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --to|--target|--push-target)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      PUSH_TARGET="$2"
      shift 2
      ;;
    --to=*|--target=*|--push-target=*)
      PUSH_TARGET="${1#*=}"
      shift
      ;;
    --path)
      [[ $# -ge 2 ]] || die "--path requires a value."
      DESTINATION_PATH="$2"
      shift 2
      ;;
    --path=*)
      DESTINATION_PATH="${1#*=}"
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
    --rclone-remote)
      [[ $# -ge 2 ]] || die "--rclone-remote requires a value."
      RCLONE_REMOTE="$2"
      shift 2
      ;;
    --rclone-remote=*)
      RCLONE_REMOTE="${1#*=}"
      shift
      ;;
    --drive-dir)
      [[ $# -ge 2 ]] || die "--drive-dir requires a value."
      DRIVE_DIR="$2"
      shift 2
      ;;
    --drive-dir=*)
      DRIVE_DIR="${1#*=}"
      shift
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || die "--remote-dir requires a value."
      REMOTE_DIR="$2"
      shift 2
      ;;
    --remote-dir=*)
      REMOTE_DIR="${1#*=}"
      shift
      ;;
    --remote-path)
      [[ $# -ge 2 ]] || die "--remote-path requires a value."
      REMOTE_PATH="$2"
      shift 2
      ;;
    --remote-path=*)
      REMOTE_PATH="${1#*=}"
      shift
      ;;
    --rclone-arg)
      [[ $# -ge 2 ]] || die "--rclone-arg requires a value."
      RCLONE_ARGS+=("$2")
      shift 2
      ;;
    --rclone-arg=*)
      RCLONE_ARGS+=("${1#*=}")
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
is_positive_int "$NUM_SHARDS" || die "--num-shards must be a positive integer."
is_positive_int "$SHARD_READY_RETRIES" || die "SHARD_READY_RETRIES must be a positive integer."
is_positive_int "$SHARD_READY_SLEEP_SECONDS" || die "SHARD_READY_SLEEP_SECONDS must be a positive integer."
[[ "${#PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "PREDICTION_TARGETS must contain at least one target."

PUSH_TARGET="${PUSH_TARGET,,}"
case "$PUSH_TARGET" in
  drive|hf)
    ;;
  *)
    die "--to/--target must be drive or hf, got: $PUSH_TARGET"
    ;;
esac

case "$NETCDF_DUPLICATE_POLICY" in
  error|first|last)
    ;;
  *)
    die "--netcdf-duplicate-policy must be error, first, or last."
    ;;
esac

case "$CREATE_REPO" in
  0|1)
    ;;
  *)
    die "CREATE_REPO must be 0 or 1, got: $CREATE_REPO"
    ;;
esac

case "$PRIVATE" in
  0|1)
    ;;
  *)
    die "PRIVATE must be 0 or 1, got: $PRIVATE"
    ;;
esac

if [[ -z "$INPUT_DIR" ]]; then
  INPUT_DIR="$RUN_PATH/eval"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$INPUT_DIR/era5_predictions_${YEAR}.nc"
fi

if [[ "${#SHARD_DIRS[@]}" -eq 0 ]]; then
  default_shard_dirs "$YEAR" "$INPUT_DIR"
fi
[[ "${#SHARD_DIRS[@]}" -eq 4 ]] || die "Exactly four shard directories are required."

BUILD_ARGS=(
  --year "$YEAR"
  --run-path "$RUN_PATH"
  --input-dir "$INPUT_DIR"
  --output-path "$OUTPUT_PATH"
  --shard-dirs "${SHARD_DIRS[@]}"
  --num-shards "$NUM_SHARDS"
  --prediction-targets "${PREDICTION_TARGET_LIST[@]}"
  --netcdf-duplicate-policy "$NETCDF_DUPLICATE_POLICY"
)

if [[ -n "$CHUNK_ROWS" ]]; then
  BUILD_ARGS+=(--chunk-rows "$CHUNK_ROWS")
fi
if [[ -n "$WRITE_TIME_CHUNK" ]]; then
  BUILD_ARGS+=(--write-time-chunk "$WRITE_TIME_CHUNK")
fi
if [[ -n "$MAX_MEMORY_GB" ]]; then
  BUILD_ARGS+=(--max-memory-gb "$MAX_MEMORY_GB")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi

PUSH_ARGS=(
  --to "$PUSH_TARGET"
  --nc "$OUTPUT_PATH"
)

if [[ -n "$DESTINATION_PATH" ]]; then
  PUSH_ARGS+=(--path "$DESTINATION_PATH")
fi
if [[ -n "$HF_REPO_ID" ]]; then
  PUSH_ARGS+=(--hf-repo-id "$HF_REPO_ID")
fi
if [[ -n "$PATH_IN_REPO" ]]; then
  PUSH_ARGS+=(--path-in-repo "$PATH_IN_REPO")
fi
if [[ -n "$REVISION" ]]; then
  PUSH_ARGS+=(--revision "$REVISION")
fi
if [[ -n "$COMMIT_MESSAGE" ]]; then
  PUSH_ARGS+=(--commit-message "$COMMIT_MESSAGE")
fi
if [[ "$PRIVATE" == "1" ]]; then
  PUSH_ARGS+=(--private)
fi
if [[ "$CREATE_REPO" == "0" ]]; then
  PUSH_ARGS+=(--skip-create-repo)
fi
if [[ -n "$RCLONE_REMOTE" ]]; then
  PUSH_ARGS+=(--rclone-remote "$RCLONE_REMOTE")
fi
if [[ -n "$DRIVE_DIR" ]]; then
  PUSH_ARGS+=(--drive-dir "$DRIVE_DIR")
fi
if [[ -n "$REMOTE_DIR" ]]; then
  PUSH_ARGS+=(--remote-dir "$REMOTE_DIR")
fi
if [[ -n "$REMOTE_PATH" ]]; then
  PUSH_ARGS+=(--remote-path "$REMOTE_PATH")
fi
for rclone_arg in "${RCLONE_ARGS[@]}"; do
  PUSH_ARGS+=(--rclone-arg "$rclone_arg")
done

echo "[$(date)] Starting ERA5 yearly post-processing."
echo "Year: $YEAR"
echo "Run path: $RUN_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Upload target: $PUSH_TARGET"
echo "Prediction targets: ${PREDICTION_TARGET_LIST[*]}"
echo "Expected rank shards per quarter: $NUM_SHARDS"
echo "Shard directories:"
printf '  %s\n' "${SHARD_DIRS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; post-processing commands would be:"
  printf '  %q' "$SCRIPT_DIR/build_year_nc.sh" "${BUILD_ARGS[@]}" --dry-run
  printf '\n'
  printf '  %q' "$SCRIPT_DIR/push_year.sh" "${PUSH_ARGS[@]}" --dry-run
  printf '\n'
  exit 0
fi

[[ -f "$SCRIPT_DIR/build_year_nc.sh" ]] || {
  echo "Build script not found: $SCRIPT_DIR/build_year_nc.sh" >&2
  exit 1
}
[[ -f "$SCRIPT_DIR/push_year.sh" ]] || {
  echo "Push script not found: $SCRIPT_DIR/push_year.sh" >&2
  exit 1
}

wait_for_shards
echo "[$(date)] All expected quarter shard files are present; building yearly NetCDF."
"$SCRIPT_DIR/build_year_nc.sh" "${BUILD_ARGS[@]}"

[[ -f "$OUTPUT_PATH" ]] || {
  echo "Built output not found: $OUTPUT_PATH" >&2
  exit 1
}

echo "[$(date)] Uploading yearly NetCDF with push_year.sh."
"$SCRIPT_DIR/push_year.sh" "${PUSH_ARGS[@]}"
