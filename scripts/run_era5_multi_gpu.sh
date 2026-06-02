#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
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

echo "[$(date)] Starting eval-era5 multi-GPU job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-local}"
echo "stdout: /scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.out"
echo "stderr: /scratch/l/luislara/EcoPerceiver/logs/eval_era5_multi_gpu.error"

RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
DB_PATH="/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db"
INITIAL_DATE="2017-06-01"
FINAL_DATE="2017-06-30"
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_CSV="$RUN_PATH/eval/era5_predictions_${DATE_TAG}.csv"
SHARD_DIR="$RUN_PATH/eval/.era5_predictions_${DATE_TAG}_multi_gpu_shards"
IGBP_EXCLUDED=(WAT SNO BSV URB CRO CVM)
PREDICTION_TARGETS=(pred_NEE pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE)

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32768}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-12}"

echo "Output CSV: $OUTPUT_CSV"
echo "Shard dir: $SHARD_DIR"
echo "GPUs: 4"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Dataloader workers per GPU: $NUM_WORKERS_PER_GPU"

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
  --prefetch-factor 1 \
  --exclude-igbp "${IGBP_EXCLUDED[@]}" \
  --prediction-targets "${PREDICTION_TARGETS[@]}" \
  --gpp-solar-threshold 2.0 \
  --skip-merge
  # --max-samples 1000000 \

echo "[$(date)] GPU inference shards complete. Submitting scripts/post_processing_era5.sh on CPU."
POST_FORMAT="${POST_FORMAT:-csv}"
POST_JOB_ID="$(sbatch --parsable --export=ALL,POST_FORMAT="$POST_FORMAT" scripts/post_processing_era5.sh)"
echo "[$(date)] Submitted ERA5 CPU post-processing job: $POST_JOB_ID (format: $POST_FORMAT)"
