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

usage() {
  cat >&2 <<EOF
Usage: $0 YEAR [--parallel|--sequential] [options]

Runs one calendar year of ERA5 inference from a two-year SQLite database.
By default, odd years use the previous even-year database:
  2016 -> <data-root>/2016_2017/era5_2016_2017.db
  2017 -> <data-root>/2016_2017/era5_2016_2017.db

Options:
  --year YEAR                 Year to infer. Positional YEAR is also accepted.
  --db-path PATH              Explicit ERA5 SQLite database path.
  --db-start-year YEAR        First year in the two-year database.
  --db-label LABEL            Database label, for example 2016_2017.
  --data-root PATH            Root directory containing date-range DB folders.
  --run-path PATH             EcoPerceiver run directory.
  --checkpoint-path PATH      Checkpoint path relative to run-path, or absolute.
  --prediction-targets LIST   Inference prediction targets, comma or space separated.
  --post-process-prediction-targets LIST
                              Yearly NetCDF prediction targets. Default: POST_PROCESS_PREDICTION_TARGETS or pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE.
  --num-shards N              Expected rank CSV shards per quarter for post-processing. Default: NUM_SHARDS or 4.
  --parallel                  Submit all quarters immediately. Default.
  --sequential                Chain quarter jobs with afterok dependencies.
  --post-process              Submit yearly NetCDF build/upload post-processing. Default.
  --skip-post-processing      Only submit inference shard jobs.
  --post-output-path PATH     Yearly NetCDF output path for post-processing.
  --overwrite-year-nc         Replace an existing yearly NetCDF during post-processing.
  --to TARGET                 Upload target for post-processing: drive or hf. Default: drive.
  --target TARGET             Alias for --to.
  --push-target TARGET        Alias for --to.
  --path PATH                 Destination path passed to push_year.sh.
  --rclone-remote REMOTE      rclone remote for Drive uploads.
  --drive-dir PATH            Google Drive folder/path for Drive uploads.
  --remote-dir REMOTE_DIR     Full rclone destination directory.
  --remote-path REMOTE_PATH   Full rclone destination file path.
  --rclone-arg ARG            Extra argument passed to rclone copyto. Repeatable.
  --hf-repo-id REPO_ID        Dataset repo id when using --to hf.
  --path-in-repo PATH         Destination path in the dataset when using --to hf.
  --revision REVISION         Branch/revision to upload to when using --to hf.
  --commit-message MESSAGE    Commit message for the upload when using --to hf.
  --private                   Create the dataset as private when using --to hf.
  --skip-create-repo          Do not create the dataset when using --to hf.
  --netcdf-duplicate-policy P Duplicate policy: error, first, or last. Default: last.
  --chunk-rows N              CSV rows per yearly NetCDF builder chunk.
  --write-time-chunk N        Time steps per NetCDF write slice.
  --max-memory-gb GB          Builder allocation guard. Default: 470.
  --dry-run                   Print sbatch commands without submitting.
  --no-check-db               Do not require the DB path to exist before submit.
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

append_post_process_prediction_targets() {
  local raw value
  raw="${1//,/ }"
  for value in $raw; do
    if [[ -n "$value" ]]; then
      POST_PROCESS_PREDICTION_TARGET_LIST+=("$value")
    fi
  done
}

resolve_checkpoint_path_for_check() {
  local checkpoint_path="$1"
  case "$checkpoint_path" in
    /*)
      echo "$checkpoint_path"
      ;;
    "~"|"~/"*)
      echo "${checkpoint_path/#\~/$HOME}"
      ;;
    *)
      echo "$RUN_PATH/$checkpoint_path"
      ;;
  esac
}

YEAR="${YEAR:-}"
MODE="${MODE:-parallel}"
POST_PROCESS="${POST_PROCESS:-1}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_DB="${CHECK_DB:-1}"
RUN_PATH="${RUN_PATH:-experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoint-11.pth}"
LOG_DIR="${LOG_DIR:-/scratch/l/luislara/EcoPerceiver/logs}"
DATA_ROOT="${DATA_ROOT:-/home/l/luislara/links/projects/aip-pal/luislara/ep/data}"
DB_PATH="${DB_PATH:-}"
DB_START_YEAR="${DB_START_YEAR:-}"
DB_LABEL="${DB_LABEL:-}"
PREDICTION_TARGETS_ENV="${PREDICTION_TARGETS:-pred_NEE pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE}"
PREDICTION_TARGET_LIST=()
append_prediction_targets "$PREDICTION_TARGETS_ENV"
POST_PROCESS_PREDICTION_TARGETS_ENV="${POST_PROCESS_PREDICTION_TARGETS:-pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE}"
POST_PROCESS_PREDICTION_TARGET_LIST=()
append_post_process_prediction_targets "$POST_PROCESS_PREDICTION_TARGETS_ENV"
NUM_SHARDS="${NUM_SHARDS:-4}"
POST_OUTPUT_PATH="${POST_OUTPUT_PATH:-${YEAR_OUTPUT_PATH:-${OUTPUT_PATH:-}}}"
OVERWRITE_YEAR_NC=0
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
NETCDF_DUPLICATE_POLICY="${NETCDF_DUPLICATE_POLICY:-last}"
CHUNK_ROWS="${BUILD_YEAR_CHUNK_ROWS:-}"
WRITE_TIME_CHUNK="${BUILD_YEAR_WRITE_TIME_CHUNK:-}"
MAX_MEMORY_GB="${BUILD_YEAR_MAX_MEMORY_GB:-470}"

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
    --db-path)
      [[ $# -ge 2 ]] || die "--db-path requires a value."
      DB_PATH="$2"
      shift 2
      ;;
    --db-path=*)
      DB_PATH="${1#*=}"
      shift
      ;;
    --db-start-year)
      [[ $# -ge 2 ]] || die "--db-start-year requires a value."
      DB_START_YEAR="$2"
      shift 2
      ;;
    --db-start-year=*)
      DB_START_YEAR="${1#*=}"
      shift
      ;;
    --db-label)
      [[ $# -ge 2 ]] || die "--db-label requires a value."
      DB_LABEL="$2"
      shift 2
      ;;
    --db-label=*)
      DB_LABEL="${1#*=}"
      shift
      ;;
    --data-root)
      [[ $# -ge 2 ]] || die "--data-root requires a value."
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT="${1#*=}"
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
    --checkpoint-path)
      [[ $# -ge 2 ]] || die "--checkpoint-path requires a value."
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --checkpoint-path=*)
      CHECKPOINT_PATH="${1#*=}"
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
    --post-process-prediction-targets|--post-prediction-targets)
      shift
      POST_PROCESS_PREDICTION_TARGET_LIST=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        append_post_process_prediction_targets "$1"
        shift
      done
      [[ "${#POST_PROCESS_PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "--post-process-prediction-targets requires at least one target."
      ;;
    --post-process-prediction-targets=*|--post-prediction-targets=*)
      POST_PROCESS_PREDICTION_TARGET_LIST=()
      append_post_process_prediction_targets "${1#*=}"
      [[ "${#POST_PROCESS_PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "--post-process-prediction-targets requires at least one target."
      shift
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
    --parallel)
      MODE="parallel"
      shift
      ;;
    --sequential)
      MODE="sequential"
      shift
      ;;
    --post-process)
      POST_PROCESS=1
      shift
      ;;
    --skip-post-processing|--no-post-processing)
      POST_PROCESS=0
      shift
      ;;
    --post-output-path|--year-output-path)
      [[ $# -ge 2 ]] || die "$1 requires a value."
      POST_OUTPUT_PATH="$2"
      shift 2
      ;;
    --post-output-path=*|--year-output-path=*)
      POST_OUTPUT_PATH="${1#*=}"
      shift
      ;;
    --overwrite-year-nc|--overwrite)
      OVERWRITE_YEAR_NC=1
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-check-db)
      CHECK_DB=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
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

case "$MODE" in
  parallel|sequential)
    ;;
  *)
    die "MODE must be parallel or sequential, got: $MODE"
    ;;
esac

case "${POST_PROCESS,,}" in
  1|true|yes)
    POST_PROCESS=1
    ;;
  0|false|no)
    POST_PROCESS=0
    ;;
  *)
    die "POST_PROCESS must be 1/0, true/false, or yes/no, got: $POST_PROCESS"
    ;;
esac

PUSH_TARGET="${PUSH_TARGET,,}"
case "$PUSH_TARGET" in
  drive|hf)
    ;;
  *)
    die "--to/--target must be drive or hf, got: $PUSH_TARGET"
    ;;
esac

[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || die "--num-shards must be a positive integer."
[[ "${#PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "PREDICTION_TARGETS must contain at least one target."
[[ "${#POST_PROCESS_PREDICTION_TARGET_LIST[@]}" -gt 0 ]] || die "POST_PROCESS_PREDICTION_TARGETS must contain at least one target."

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

year_number=$((10#$YEAR))
if [[ -n "$DB_START_YEAR" ]]; then
  [[ "$DB_START_YEAR" =~ ^[0-9]{4}$ ]] || die "DB_START_YEAR must be a four-digit year, got: $DB_START_YEAR"
  db_start_number=$((10#$DB_START_YEAR))
else
  if (( year_number % 2 == 0 )); then
    db_start_number="$year_number"
  else
    db_start_number=$((year_number - 1))
  fi
fi

if [[ -z "$DB_LABEL" ]]; then
  db_end_number=$((db_start_number + 1))
  DB_LABEL="${db_start_number}_${db_end_number}"
fi

if [[ -z "$DB_PATH" ]]; then
  DB_PATH="${DATA_ROOT}/${DB_LABEL}/era5_${DB_LABEL}.db"
fi

if [[ "$CHECK_DB" != "0" && ! -f "$DB_PATH" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "WARNING: ERA5 DB not found: $DB_PATH" >&2
  else
    echo "ERA5 DB not found: $DB_PATH" >&2
    echo "Use --db-path, --db-start-year, --db-label, or --data-root to point at the two-year DB." >&2
    exit 2
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  command -v sbatch >/dev/null || {
    echo "sbatch not found in PATH; cannot submit ERA5 jobs." >&2
    exit 2
  }
  [[ -f scripts/era5/run_multi_gpu.sh ]] || {
    echo "Child job script not found: scripts/era5/run_multi_gpu.sh" >&2
    exit 2
  }
  if [[ "$POST_PROCESS" == "1" ]]; then
    [[ -f scripts/era5/post_processing.sh ]] || {
      echo "Post-processing job script not found: scripts/era5/post_processing.sh" >&2
      exit 2
    }
  fi
  [[ -n "${SCRATCH:-}" ]] || {
    echo "SCRATCH is not set; scripts/era5/run_multi_gpu.sh needs it to activate the environment." >&2
    exit 2
  }
  [[ -f "$SCRATCH/env/ecoperceiver/bin/activate" ]] || {
    echo "EcoPerceiver environment activation script not found: $SCRATCH/env/ecoperceiver/bin/activate" >&2
    exit 2
  }
  [[ -d "$RUN_PATH" ]] || {
    echo "Run path not found: $RUN_PATH" >&2
    exit 2
  }
  [[ -n "$CHECKPOINT_PATH" ]] || {
    echo "CHECKPOINT_PATH must not be empty." >&2
    exit 2
  }
  checkpoint_path_for_check="$(resolve_checkpoint_path_for_check "$CHECKPOINT_PATH")"
  [[ -f "$checkpoint_path_for_check" ]] || {
    echo "Checkpoint not found: $checkpoint_path_for_check" >&2
    exit 2
  }
fi

mkdir -p "$LOG_DIR"

starts=("${YEAR}-01-01" "${YEAR}-04-01" "${YEAR}-07-01" "${YEAR}-10-01")
ends=("${YEAR}-03-31" "${YEAR}-06-30" "${YEAR}-09-30" "${YEAR}-12-31")

PREDICTION_TARGETS_VALUE="${PREDICTION_TARGET_LIST[*]}"
POST_PROCESS_PREDICTION_TARGETS_VALUE="${POST_PROCESS_PREDICTION_TARGET_LIST[*]}"
export PREDICTION_TARGETS="$PREDICTION_TARGETS_VALUE"
export POST_PROCESS_PREDICTION_TARGETS="$POST_PROCESS_PREDICTION_TARGETS_VALUE"
export CHECKPOINT_PATH

previous_dependency_id=""
dependency_ids=()
year_shard_dirs=()
echo "Submitting ERA5 one-year inference for $YEAR in $MODE mode."
echo "Two-year DB label: $DB_LABEL"
echo "DB path: $DB_PATH"
echo "Run path: $RUN_PATH"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Inference prediction targets: $PREDICTION_TARGETS_VALUE"
if [[ "$POST_PROCESS" == "1" ]]; then
  echo "Post-processing: enabled after all four quarter jobs finish successfully."
  echo "Post-processing upload target: $PUSH_TARGET"
  echo "Post-processing prediction targets: $POST_PROCESS_PREDICTION_TARGETS_VALUE"
  echo "Post-processing expected rank shards per quarter: $NUM_SHARDS"
  if [[ -n "$POST_OUTPUT_PATH" ]]; then
    echo "Post-processing yearly output path: $POST_OUTPUT_PATH"
  fi
else
  echo "Post-processing: disabled."
fi

for chunk_index in "${!starts[@]}"; do
  initial_date="${starts[$chunk_index]}"
  final_date="${ends[$chunk_index]}"
  date_tag="${initial_date//-/}_to_${final_date//-/}"
  job_name="run-mgpu-${YEAR}-q$((chunk_index + 1))"
  log_base="$LOG_DIR/run_multi_gpu_${date_tag}"
  log_out="${log_base}.out"
  log_err="${log_base}.error"
  shard_dir="$RUN_PATH/eval/.era5_predictions_${date_tag}_multi_gpu_shards"
  year_shard_dirs+=("$shard_dir")

  sbatch_args=(
    --parsable
    --job-name "$job_name"
    --output "$log_out"
    --error "$log_err"
    --export=ALL,RUN_PATH="$RUN_PATH",DB_PATH="$DB_PATH",INITIAL_DATE="$initial_date",FINAL_DATE="$final_date",LOG_DIR="$LOG_DIR",GPU_LOG_OUT="$log_out",GPU_LOG_ERR="$log_err",SHARD_DIR="$shard_dir"
  )

  if [[ "$MODE" == "sequential" && -n "$previous_dependency_id" ]]; then
    sbatch_args+=(--dependency="afterok:$previous_dependency_id")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: PREDICTION_TARGETS=%q CHECKPOINT_PATH=%q sbatch' "$PREDICTION_TARGETS" "$CHECKPOINT_PATH"
    printf ' %q' "${sbatch_args[@]}" scripts/era5/run_multi_gpu.sh
    printf '\n'
    job_id="dryrun-$((chunk_index + 1))"
  else
    job_id="$(sbatch "${sbatch_args[@]}" scripts/era5/run_multi_gpu.sh)"
  fi

  dependency_id="${job_id%%;*}"
  dependency_ids+=("$dependency_id")
  echo "Chunk $((chunk_index + 1)): $initial_date to $final_date -> $job_id"
  previous_dependency_id="$dependency_id"
done

if [[ "$POST_PROCESS" == "1" ]]; then
  [[ "${#dependency_ids[@]}" -eq 4 ]] || die "Expected four quarter job IDs for post-processing dependency."
  [[ "${#year_shard_dirs[@]}" -eq 4 ]] || die "Expected four quarter shard directories for post-processing."

  dependency_list="$(IFS=:; echo "${dependency_ids[*]}")"
  post_dependency="afterok:$dependency_list"
  post_job_name="post-era5-${YEAR}"
  post_log_base="$LOG_DIR/post_processing_${YEAR}"
  post_log_out="${post_log_base}.out"
  post_log_err="${post_log_base}.error"

  post_args=(
    --year "$YEAR"
    --run-path "$RUN_PATH"
    --input-dir "$RUN_PATH/eval"
    --shard-dirs "${year_shard_dirs[@]}"
    --num-shards "$NUM_SHARDS"
    --prediction-targets "${POST_PROCESS_PREDICTION_TARGET_LIST[@]}"
    --netcdf-duplicate-policy "$NETCDF_DUPLICATE_POLICY"
    --to "$PUSH_TARGET"
  )

  if [[ -n "$POST_OUTPUT_PATH" ]]; then
    post_args+=(--output-path "$POST_OUTPUT_PATH")
  fi
  if [[ -n "$DESTINATION_PATH" ]]; then
    post_args+=(--path "$DESTINATION_PATH")
  fi
  if [[ -n "$RCLONE_REMOTE" ]]; then
    post_args+=(--rclone-remote "$RCLONE_REMOTE")
  fi
  if [[ -n "$DRIVE_DIR" ]]; then
    post_args+=(--drive-dir "$DRIVE_DIR")
  fi
  if [[ -n "$REMOTE_DIR" ]]; then
    post_args+=(--remote-dir "$REMOTE_DIR")
  fi
  if [[ -n "$REMOTE_PATH" ]]; then
    post_args+=(--remote-path "$REMOTE_PATH")
  fi
  for rclone_arg in "${RCLONE_ARGS[@]}"; do
    post_args+=(--rclone-arg "$rclone_arg")
  done
  if [[ -n "$HF_REPO_ID" ]]; then
    post_args+=(--hf-repo-id "$HF_REPO_ID")
  fi
  if [[ -n "$PATH_IN_REPO" ]]; then
    post_args+=(--path-in-repo "$PATH_IN_REPO")
  fi
  if [[ -n "$REVISION" ]]; then
    post_args+=(--revision "$REVISION")
  fi
  if [[ -n "$COMMIT_MESSAGE" ]]; then
    post_args+=(--commit-message "$COMMIT_MESSAGE")
  fi
  if [[ -n "$CHUNK_ROWS" ]]; then
    post_args+=(--chunk-rows "$CHUNK_ROWS")
  fi
  if [[ -n "$WRITE_TIME_CHUNK" ]]; then
    post_args+=(--write-time-chunk "$WRITE_TIME_CHUNK")
  fi
  if [[ -n "$MAX_MEMORY_GB" ]]; then
    post_args+=(--max-memory-gb "$MAX_MEMORY_GB")
  fi
  if [[ "$OVERWRITE_YEAR_NC" == "1" ]]; then
    post_args+=(--overwrite)
  fi
  if [[ "$PRIVATE" == "1" ]]; then
    post_args+=(--private)
  fi
  if [[ "$CREATE_REPO" == "0" ]]; then
    post_args+=(--skip-create-repo)
  fi

  post_sbatch_args=(
    --parsable
    --job-name "$post_job_name"
    --output "$post_log_out"
    --error "$post_log_err"
    --dependency="$post_dependency"
    --export=ALL,RUN_PATH="$RUN_PATH",LOG_DIR="$LOG_DIR",POST_LOG_OUT="$post_log_out",POST_LOG_ERR="$post_log_err"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: sbatch'
    printf ' %q' "${post_sbatch_args[@]}" scripts/era5/post_processing.sh "${post_args[@]}"
    printf '\n'
    post_job_id="dryrun-post"
  else
    post_job_id="$(sbatch "${post_sbatch_args[@]}" scripts/era5/post_processing.sh "${post_args[@]}")"
  fi

  echo "Post-processing dependency: $post_dependency"
  echo "Post-processing job: $post_job_id"
fi
