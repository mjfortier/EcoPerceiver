#!/bin/bash

set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 YEAR [--parallel|--sequential] [--post-format csv|netcdf] [options]

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
  --prediction-targets LIST   Prediction targets, comma or space separated.
  --parallel                  Submit all quarters immediately. Default.
  --sequential                Chain quarter jobs with afterok dependencies.
  --post-format csv|netcdf    Final post-processing output format. Default: netcdf.
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

YEAR="${YEAR:-}"
MODE="${MODE:-parallel}"
POST_FORMAT="${POST_FORMAT:-netcdf}"
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
    --parallel)
      MODE="parallel"
      shift
      ;;
    --sequential)
      MODE="sequential"
      shift
      ;;
    --post-format)
      [[ $# -ge 2 ]] || die "--post-format requires a value."
      POST_FORMAT="$2"
      shift 2
      ;;
    --post-format=*)
      POST_FORMAT="${1#*=}"
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

case "$POST_FORMAT" in
  csv|netcdf)
    ;;
  *)
    die "POST_FORMAT must be csv or netcdf, got: $POST_FORMAT"
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

mkdir -p "$LOG_DIR"

starts=("${YEAR}-01-01" "${YEAR}-04-01" "${YEAR}-07-01" "${YEAR}-10-01")
ends=("${YEAR}-03-31" "${YEAR}-06-30" "${YEAR}-09-30" "${YEAR}-12-31")

PREDICTION_TARGETS_VALUE="${PREDICTION_TARGET_LIST[*]}"
export PREDICTION_TARGETS="$PREDICTION_TARGETS_VALUE"
export CHECKPOINT_PATH

previous_dependency_id=""
echo "Submitting ERA5 one-year inference for $YEAR in $MODE mode."
echo "Two-year DB label: $DB_LABEL"
echo "DB path: $DB_PATH"
echo "Run path: $RUN_PATH"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Post format: $POST_FORMAT"
echo "Prediction targets: $PREDICTION_TARGETS_VALUE"

for chunk_index in "${!starts[@]}"; do
  initial_date="${starts[$chunk_index]}"
  final_date="${ends[$chunk_index]}"
  date_tag="${initial_date//-/}_to_${final_date//-/}"
  job_name="run-mgpu-${YEAR}-q$((chunk_index + 1))"
  log_base="$LOG_DIR/run_multi_gpu_${date_tag}"
  log_out="${log_base}.out"
  log_err="${log_base}.error"
  output_csv="$RUN_PATH/eval/era5_predictions_${date_tag}.csv"
  shard_dir="$RUN_PATH/eval/.era5_predictions_${date_tag}_multi_gpu_shards"

  sbatch_args=(
    --parsable
    --job-name "$job_name"
    --output "$log_out"
    --error "$log_err"
    --export=ALL,RUN_PATH="$RUN_PATH",DB_PATH="$DB_PATH",INITIAL_DATE="$initial_date",FINAL_DATE="$final_date",POST_FORMAT="$POST_FORMAT",LOG_DIR="$LOG_DIR",GPU_LOG_OUT="$log_out",GPU_LOG_ERR="$log_err",OUTPUT_CSV="$output_csv",SHARD_DIR="$shard_dir",OUTPUT_PATH=
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
  previous_dependency_id="$dependency_id"
  echo "Chunk $((chunk_index + 1)): $initial_date to $final_date -> $job_id"
done
