#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=48
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=eval-era5-mgpu
#SBATCH --account=aip-pal

set -euo pipefail

source "$SCRATCH/env/ecoperceiver/bin/activate"
cd ~/links/scratch/EcoPerceiver

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

GPU_LOG_OUT="${GPU_LOG_OUT:-/scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.out}"
GPU_LOG_ERR="${GPU_LOG_ERR:-/scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.error}"

echo "[$(date)] Starting eval-era5 multi-GPU job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-local}"
echo "stdout: $GPU_LOG_OUT"
echo "stderr: $GPU_LOG_ERR"

RUN_PATH="${RUN_PATH:-experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0}"
DB_PATH="${DB_PATH:-/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db}"
INITIAL_DATE="${INITIAL_DATE:-2017-06-01}"
FINAL_DATE="${FINAL_DATE:-2017-06-30}"
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_CSV="${OUTPUT_CSV:-$RUN_PATH/eval/era5_predictions_${DATE_TAG}.csv}"
SHARD_DIR="${SHARD_DIR:-$RUN_PATH/eval/.era5_predictions_${DATE_TAG}_multi_gpu_shards}"
LOG_DIR="${LOG_DIR:-/scratch/l/luislara/EcoPerceiver/logs}"
IGBP_EXCLUDED=(WAT SNO BSV URB CRO CVM)
PREDICTION_TARGETS=(pred_NEE pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE)

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

echo "Date window: $INITIAL_DATE to $FINAL_DATE"
echo "Output CSV: $OUTPUT_CSV"
echo "Shard dir: $SHARD_DIR"
echo "GPUs: 4"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Dataloader workers per GPU: $NUM_WORKERS_PER_GPU"
echo "Prefetch factor: $PREFETCH_FACTOR"
echo "Dataloader in-order delivery: $DATALOADER_IN_ORDER_LABEL"
echo "Temporary shard order key: __sample_order (dropped during post-processing)"
echo "Distributed timeout minutes: $DIST_TIMEOUT_MINUTES"

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  eval/test_era5_multi_gpu.py \
  --run-path "$RUN_PATH" \
  --checkpoint-path checkpoint-11.pth \
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
  --prediction-targets "${PREDICTION_TARGETS[@]}" \
  --gpp-solar-threshold 2.0 \
  --skip-merge
  # --max-samples 1000000 \

echo "[$(date)] GPU inference shards complete. Submitting scripts/post_processing_era5.sh on CPU."
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
    --job-name "post-era5-${DATE_TAG}" \
    --output "$LOG_DIR/post_processing_era5_${DATE_TAG}.out" \
    --error "$LOG_DIR/post_processing_era5_${DATE_TAG}.error" \
    --export=ALL,RUN_PATH="$RUN_PATH",INITIAL_DATE="$INITIAL_DATE",FINAL_DATE="$FINAL_DATE",POST_FORMAT="$POST_FORMAT",SHARD_DIR="$SHARD_DIR",OUTPUT_PATH="$POST_OUTPUT_PATH" \
    scripts/post_processing_era5.sh
)"
echo "[$(date)] Submitted ERA5 CPU post-processing job: $POST_JOB_ID (format: $POST_FORMAT, output: $POST_OUTPUT_PATH)"
