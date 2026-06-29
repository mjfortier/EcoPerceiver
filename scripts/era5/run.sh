#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/run.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/run.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=run
#SBATCH --account=aip-pal

set -euo pipefail

source $SCRATCH/env/ecoperceiver/bin/activate
cd ~/links/scratch/EcoPerceiver

export PYTHONUNBUFFERED=1
echo "[$(date)] Starting single-process inference job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-local}"
echo "stdout: /scratch/l/luislara/EcoPerceiver/logs/run.out"
echo "stderr: /scratch/l/luislara/EcoPerceiver/logs/run.error"

RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
DB_PATH="/home/l/luislara/links/projects/aip-pal/luislara/ep/data/era5.db"
INITIAL_DATE="2017-06-01"
FINAL_DATE="2017-06-30"
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_CSV="$RUN_PATH/eval/era5_predictions_${DATE_TAG}_single.csv"
IGBP_EXCLUDED=(WAT SNO BSV URB CRO CVM)
PREDICTION_TARGETS=(pred_GPP_DT pred_RECO_DT pred_FCH4 pred_LE)

echo "Output CSV: $OUTPUT_CSV"

python3 -u eval/era5/test_era5.py \
  --run-path "$RUN_PATH" \
  --checkpoint-path checkpoint-11.pth \
  --db-path "$DB_PATH" \
  --initial-date "$INITIAL_DATE" \
  --final-date "$FINAL_DATE" \
  --output-csv "$OUTPUT_CSV" \
  --batch-size 32768 \
  --num-workers 16 \
  --prefetch-factor 1 \
  --exclude-igbp "${IGBP_EXCLUDED[@]}" \
  --prediction-targets "${PREDICTION_TARGETS[@]}" \
  --gpp-solar-threshold 2.0 \
  --max-samples 1000000
