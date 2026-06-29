#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/merge_prediction_shards.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/merge_prediction_shards.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=merge-shards
#SBATCH --account=aip-pal

set -euo pipefail

source "$SCRATCH/env/ecoperceiver/bin/activate"
cd ~/links/scratch/EcoPerceiver

export PYTHONUNBUFFERED=1

usage() {
  echo "Usage: $0 [--format csv|netcdf] [--prediction-targets TARGET ...] [--sort-output|--no-sort-output] [--netcdf-duplicate-policy error|first|last|mean]" >&2
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

POST_FORMAT="${POST_FORMAT:-netcdf}"
SORT_OUTPUT="${SORT_OUTPUT:-1}"
NUM_SHARDS="${NUM_SHARDS:-4}"
NETCDF_DUPLICATE_POLICY="${NETCDF_DUPLICATE_POLICY:-last}"
PREDICTION_TARGETS_ENV="${PREDICTION_TARGETS:-pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE}"
PREDICTION_TARGET_LIST=()
if [[ -n "$PREDICTION_TARGETS_ENV" ]]; then
  append_prediction_targets "$PREDICTION_TARGETS_ENV"
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      if [[ $# -lt 2 ]]; then
        echo "--format requires a value." >&2
        usage
        exit 2
      fi
      POST_FORMAT="$2"
      shift 2
      ;;
    --format=*)
      POST_FORMAT="${1#*=}"
      shift
      ;;
    --prediction-targets)
      shift
      PREDICTION_TARGET_LIST=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        append_prediction_targets "$1"
        shift
      done
      if [[ "${#PREDICTION_TARGET_LIST[@]}" -eq 0 ]]; then
        echo "--prediction-targets requires at least one target." >&2
        usage
        exit 2
      fi
      ;;
    --prediction-targets=*)
      PREDICTION_TARGET_LIST=()
      append_prediction_targets "${1#*=}"
      if [[ "${#PREDICTION_TARGET_LIST[@]}" -eq 0 ]]; then
        echo "--prediction-targets requires at least one target." >&2
        usage
        exit 2
      fi
      shift
      ;;
    --sort-output)
      SORT_OUTPUT=1
      shift
      ;;
    --no-sort-output)
      SORT_OUTPUT=0
      shift
      ;;
    --netcdf-duplicate-policy)
      if [[ $# -lt 2 ]]; then
        echo "--netcdf-duplicate-policy requires a value." >&2
        usage
        exit 2
      fi
      NETCDF_DUPLICATE_POLICY="$2"
      shift 2
      ;;
    --netcdf-duplicate-policy=*)
      NETCDF_DUPLICATE_POLICY="${1#*=}"
      shift
      ;;
    csv|netcdf)
      POST_FORMAT="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$POST_FORMAT" in
  csv)
    OUTPUT_EXT="csv"
    ;;
  netcdf)
    OUTPUT_EXT="nc"
    ;;
  *)
    echo "POST_FORMAT must be csv or netcdf, got: $POST_FORMAT" >&2
    exit 2
    ;;
esac

case "$NETCDF_DUPLICATE_POLICY" in
  error|first|last|mean)
    ;;
  *)
    echo "NETCDF_DUPLICATE_POLICY must be error, first, last, or mean; got: $NETCDF_DUPLICATE_POLICY" >&2
    exit 2
    ;;
esac

if [[ ! "$NUM_SHARDS" =~ ^[0-9]+$ || "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be a positive integer, got: $NUM_SHARDS" >&2
  exit 2
fi

case "${SORT_OUTPUT,,}" in
  0|false|no)
    SORT_ARGS=()
    SORT_LABEL="false"
    ;;
  1|true|yes)
    SORT_ARGS=(--sort-output)
    SORT_LABEL="true"
    ;;
  *)
    echo "SORT_OUTPUT must be 0/1, false/true, or no/yes; got: $SORT_OUTPUT" >&2
    exit 2
    ;;
esac

if [[ -n "${SORT_TMP_DIR:-}" ]]; then
  SORT_ARGS+=(--sort-tmp-dir "$SORT_TMP_DIR")
fi

PREDICTION_TARGET_ARGS=()
if [[ "${#PREDICTION_TARGET_LIST[@]}" -gt 0 ]]; then
  PREDICTION_TARGET_ARGS=(--prediction-targets "${PREDICTION_TARGET_LIST[@]}")
  PREDICTION_TARGET_LABEL="${PREDICTION_TARGET_LIST[*]}"
else
  PREDICTION_TARGET_LABEL="<all shard prediction columns>"
fi

RUN_PATH="${RUN_PATH:-experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0}"
INITIAL_DATE="${INITIAL_DATE:-2017-06-01}"
FINAL_DATE="${FINAL_DATE:-2017-06-30}"
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_PATH="${OUTPUT_PATH:-$RUN_PATH/eval/era5_predictions_${DATE_TAG}.${OUTPUT_EXT}}"
SHARD_DIR="${SHARD_DIR:-$RUN_PATH/eval/.era5_predictions_${DATE_TAG}_multi_gpu_shards}"

echo "[$(date)] Starting prediction-shard merge job ${SLURM_JOB_ID:-local}"
echo "Format: $POST_FORMAT"
echo "Date window: $INITIAL_DATE to $FINAL_DATE"
echo "Shard dir: $SHARD_DIR"
echo "Output path: $OUTPUT_PATH"
echo "Num shards: $NUM_SHARDS"
echo "Prediction targets: $PREDICTION_TARGET_LABEL"
echo "Sort output: $SORT_LABEL"
echo "NetCDF duplicate policy: $NETCDF_DUPLICATE_POLICY"
echo "Temporary shard order key: __sample_order (dropped from final output when present)"

python3 -u eval/era5/merge_era5_shards.py \
  --format "$POST_FORMAT" \
  --shard-dir "$SHARD_DIR" \
  --output-path "$OUTPUT_PATH" \
  --num-shards "$NUM_SHARDS" \
  "${SORT_ARGS[@]}" \
  --netcdf-duplicate-policy "$NETCDF_DUPLICATE_POLICY" \
  "${PREDICTION_TARGET_ARGS[@]}"
  # --cleanup
