#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=48
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/run_multi_gpu.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/run_multi_gpu.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=run-mgpu
#SBATCH --account=aip-pal

set -euo pipefail

source "$SCRATCH/env/ecoperceiver/bin/activate"
cd ~/links/scratch/EcoPerceiver

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

append_prediction_targets() {
  local raw value
  raw="${1//,/ }"
  for value in $raw; do
    if [[ -n "$value" ]]; then
      PREDICTION_TARGET_LIST+=("$value")
    fi
  done
}

GPU_LOG_OUT="${GPU_LOG_OUT:-/scratch/l/luislara/EcoPerceiver/logs/run_multi_gpu.out}"
GPU_LOG_ERR="${GPU_LOG_ERR:-/scratch/l/luislara/EcoPerceiver/logs/run_multi_gpu.error}"

echo "[$(date)] Starting date-range multi-GPU inference job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-local}"
echo "stdout: $GPU_LOG_OUT"
echo "stderr: $GPU_LOG_ERR"

RUN_PATH="${RUN_PATH:-experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoint-11.pth}"
INITIAL_DATE="${INITIAL_DATE:-2017-06-01}"
FINAL_DATE="${FINAL_DATE:-2017-06-30}"
DATA_ROOT="${DATA_ROOT:-/home/l/luislara/links/projects/aip-pal/luislara/ep/data}"
DB_PATH="${DB_PATH:-}"
DB_START_YEAR="${DB_START_YEAR:-}"
DB_LABEL="${DB_LABEL:-}"
if [[ -z "$DB_PATH" ]]; then
  initial_year="${INITIAL_DATE:0:4}"
  if [[ ! "$initial_year" =~ ^[0-9]{4}$ ]]; then
    echo "Cannot infer DB_PATH from INITIAL_DATE=$INITIAL_DATE; set DB_PATH explicitly." >&2
    exit 2
  fi
  if [[ -z "$DB_LABEL" ]]; then
    if [[ -n "$DB_START_YEAR" ]]; then
      if [[ ! "$DB_START_YEAR" =~ ^[0-9]{4}$ ]]; then
        echo "DB_START_YEAR must be a four-digit year, got: $DB_START_YEAR" >&2
        exit 2
      fi
      db_start_number=$((10#$DB_START_YEAR))
    else
      initial_year_number=$((10#$initial_year))
      if (( initial_year_number % 2 == 0 )); then
        db_start_number="$initial_year_number"
      else
        db_start_number=$((initial_year_number - 1))
      fi
    fi
    DB_LABEL="${db_start_number}_$((db_start_number + 1))"
  fi
  DB_PATH="${DATA_ROOT}/${DB_LABEL}/era5_${DB_LABEL}.db"
fi
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_CSV="${OUTPUT_CSV:-$RUN_PATH/eval/era5_predictions_${DATE_TAG}.csv}"
SHARD_DIR="${SHARD_DIR:-$RUN_PATH/eval/.era5_predictions_${DATE_TAG}_multi_gpu_shards}"
LOG_DIR="${LOG_DIR:-/scratch/l/luislara/EcoPerceiver/logs}"
IGBP_EXCLUDED=(WAT SNO BSV URB CRO CVM)
PREDICTION_TARGETS_ENV="${PREDICTION_TARGETS:-pred_NEE pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE}"
PREDICTION_TARGET_LIST=(pred_NEE pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE)
if [[ -n "$PREDICTION_TARGETS_ENV" ]]; then
  PREDICTION_TARGET_LIST=()
  append_prediction_targets "$PREDICTION_TARGETS_ENV"
fi
PREDICTION_TARGETS_VALUE="${PREDICTION_TARGET_LIST[*]}"
export PREDICTION_TARGETS="$PREDICTION_TARGETS_VALUE"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32768}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-120}"
DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-0}"

case "${DATALOADER_IN_ORDER,,}" in
  0|false|no)
    DATALOADER_ORDER_ARGS=(--dataloader-out-of-order)
    DATALOADER_IN_ORDER_LABEL="false"
    ;;
  1|true|yes)
    DATALOADER_ORDER_ARGS=()
    DATALOADER_IN_ORDER_LABEL="true"
    ;;
  *)
    echo "DATALOADER_IN_ORDER must be 0/1, false/true, or no/yes; got: $DATALOADER_IN_ORDER" >&2
    exit 2
    ;;
esac

if [[ ! -f "$DB_PATH" ]]; then
  echo "ERA5 DB not found: $DB_PATH" >&2
  exit 2
fi

echo "Date window: $INITIAL_DATE to $FINAL_DATE"
echo "DB path: $DB_PATH"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Output CSV: $OUTPUT_CSV"
echo "Shard dir: $SHARD_DIR"
echo "GPUs: 4"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Dataloader workers per GPU: $NUM_WORKERS_PER_GPU"
echo "Prefetch factor: $PREFETCH_FACTOR"
echo "Dataloader in-order delivery: $DATALOADER_IN_ORDER_LABEL"
echo "Prediction targets: $PREDICTION_TARGETS_VALUE"
echo "Temporary shard order key: __sample_order (dropped during post-processing)"
echo "Distributed timeout minutes: $DIST_TIMEOUT_MINUTES"

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  eval/test_era5_multi_gpu.py \
  --run-path "$RUN_PATH" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --db-path "$DB_PATH" \
  --initial-date "$INITIAL_DATE" \
  --final-date "$FINAL_DATE" \
  --output-csv "$OUTPUT_CSV" \
  --shard-dir "$SHARD_DIR" \
  --batch-size "$BATCH_SIZE_PER_GPU" \
  --num-workers "$NUM_WORKERS_PER_GPU" \
  --distributed-timeout-minutes "$DIST_TIMEOUT_MINUTES" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  "${DATALOADER_ORDER_ARGS[@]}" \
  --exclude-igbp "${IGBP_EXCLUDED[@]}" \
  --prediction-targets "${PREDICTION_TARGET_LIST[@]}" \
  --gpp-solar-threshold 2.0 \
  --skip-merge
  # --max-samples 1000000 \

echo "[$(date)] GPU inference shards complete. Submitting scripts/era5/merge_prediction_shards.sh on CPU."
POST_FORMAT="${POST_FORMAT:-csv}"
case "$POST_FORMAT" in
  csv)
    POST_OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_CSV}"
    ;;
  netcdf)
    POST_OUTPUT_PATH="${OUTPUT_PATH:-$RUN_PATH/eval/era5_predictions_${DATE_TAG}.nc}"
    ;;
  *)
    echo "POST_FORMAT must be csv or netcdf, got: $POST_FORMAT" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
POST_JOB_ID="$(
  sbatch --parsable \
    --job-name "merge-shards-${DATE_TAG}" \
    --output "$LOG_DIR/merge_prediction_shards_${DATE_TAG}.out" \
    --error "$LOG_DIR/merge_prediction_shards_${DATE_TAG}.error" \
    --export=ALL,RUN_PATH="$RUN_PATH",INITIAL_DATE="$INITIAL_DATE",FINAL_DATE="$FINAL_DATE",POST_FORMAT="$POST_FORMAT",SHARD_DIR="$SHARD_DIR",OUTPUT_PATH="$POST_OUTPUT_PATH" \
    scripts/era5/merge_prediction_shards.sh
)"
echo "[$(date)] Submitted ERA5 CPU post-processing job: $POST_JOB_ID (format: $POST_FORMAT, output: $POST_OUTPUT_PATH)"
