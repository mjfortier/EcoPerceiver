#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 YEAR [--parallel|--sequential] [--post-format csv|netcdf] [--dry-run]" >&2
  echo "Example: $0 2017" >&2
}

YEAR="${YEAR:-}"
MODE="${MODE:-parallel}"
POST_FORMAT="${POST_FORMAT:-csv}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PATH="${RUN_PATH:-experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0}"
LOG_DIR="${LOG_DIR:-/scratch/l/luislara/EcoPerceiver/logs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --year)
      YEAR="$2"
      shift 2
      ;;
    --year=*)
      YEAR="${1#*=}"
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$YEAR" ]]; then
        YEAR="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

YEAR="${YEAR:-2017}"

if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
  echo "YEAR must be a four-digit year, got: ${YEAR:-<empty>}" >&2
  usage
  exit 2
fi

case "$MODE" in
  parallel|sequential)
    ;;
  *)
    echo "MODE must be parallel or sequential, got: $MODE" >&2
    exit 2
    ;;
esac

case "$POST_FORMAT" in
  csv|netcdf)
    ;;
  *)
    echo "POST_FORMAT must be csv or netcdf, got: $POST_FORMAT" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

starts=("${YEAR}-01-01" "${YEAR}-04-01" "${YEAR}-07-01" "${YEAR}-10-01")
ends=("${YEAR}-03-31" "${YEAR}-06-30" "${YEAR}-09-30" "${YEAR}-12-31")

previous_dependency_id=""
echo "Submitting ERA5 one-year inference for $YEAR in $MODE mode."

for chunk_index in "${!starts[@]}"; do
  initial_date="${starts[$chunk_index]}"
  final_date="${ends[$chunk_index]}"
  date_tag="${initial_date//-/}_to_${final_date//-/}"
  job_name="eval-era5-${YEAR}-$((chunk_index + 1))"
  log_base="$LOG_DIR/eval_era5_multi_gpu_${date_tag}"
  log_out="${log_base}.out"
  log_err="${log_base}.error"
  output_csv="$RUN_PATH/eval/era5_predictions_${date_tag}.csv"
  shard_dir="$RUN_PATH/eval/.era5_predictions_${date_tag}_multi_gpu_shards"

  sbatch_args=(
    --parsable
    --job-name "$job_name"
    --output "$log_out"
    --error "$log_err"
    --export=ALL,RUN_PATH="$RUN_PATH",INITIAL_DATE="$initial_date",FINAL_DATE="$final_date",POST_FORMAT="$POST_FORMAT",LOG_DIR="$LOG_DIR",GPU_LOG_OUT="$log_out",GPU_LOG_ERR="$log_err",OUTPUT_CSV="$output_csv",SHARD_DIR="$shard_dir",OUTPUT_PATH=
  )

  if [[ "$MODE" == "sequential" && -n "$previous_dependency_id" ]]; then
    sbatch_args+=(--dependency="afterok:$previous_dependency_id")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: sbatch'
    printf ' %q' "${sbatch_args[@]}" scripts/run_era5_multi_gpu.sh
    printf '\n'
    job_id="dryrun-$((chunk_index + 1))"
  else
    job_id="$(sbatch "${sbatch_args[@]}" scripts/run_era5_multi_gpu.sh)"
  fi

  dependency_id="${job_id%%;*}"
  previous_dependency_id="$dependency_id"
  echo "Chunk $((chunk_index + 1)): $initial_date to $final_date -> $job_id"
done
