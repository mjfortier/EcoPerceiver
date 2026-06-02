#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/l/luislara/EcoPerceiver/logs/post_processing_era5.out
#SBATCH --error=/scratch/l/luislara/EcoPerceiver/logs/post_processing_era5.error
#SBATCH --open-mode=truncate
#SBATCH --job-name=post-era5
#SBATCH --account=aip-pal

set -euo pipefail

source "$SCRATCH/env/ecoperceiver/bin/activate"
cd ~/links/scratch/EcoPerceiver

export PYTHONUNBUFFERED=1

POST_FORMAT="${POST_FORMAT:-csv}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      POST_FORMAT="$2"
      shift 2
      ;;
    --format=*)
      POST_FORMAT="${1#*=}"
      shift
      ;;
    csv|netcdf)
      POST_FORMAT="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--format csv|netcdf]" >&2
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

RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"
INITIAL_DATE="2017-06-01"
FINAL_DATE="2017-06-30"
DATE_TAG="${INITIAL_DATE//-/}_to_${FINAL_DATE//-/}"
OUTPUT_PATH="$RUN_PATH/eval/era5_predictions_${DATE_TAG}.${OUTPUT_EXT}"
SHARD_DIR="$RUN_PATH/eval/.era5_predictions_${DATE_TAG}_multi_gpu_shards"

echo "[$(date)] Starting ERA5 post-processing job ${SLURM_JOB_ID:-local}"
echo "Format: $POST_FORMAT"
echo "Shard dir: $SHARD_DIR"
echo "Output path: $OUTPUT_PATH"

python3 -u eval/merge_era5_shards.py \
  --format "$POST_FORMAT" \
  --shard-dir "$SHARD_DIR" \
  --output-path "$OUTPUT_PATH" \
  --num-shards 4 
  # --cleanup
