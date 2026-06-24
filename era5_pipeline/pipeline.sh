#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/l/luislara/links/scratch/EcoPerceiver/era5_pipeline/logs/pipeline_%j.out
#SBATCH --error=/home/l/luislara/links/scratch/EcoPerceiver/era5_pipeline/logs/pipeline_%j.error
#SBATCH --job-name=era5-pipeline
#SBATCH --account=aip-pal

set -euo pipefail

source "$SCRATCH/env/ecoperceiver/bin/activate"
cd "$HOME/links/scratch/EcoPerceiver"
mkdir -p era5_pipeline/logs

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u era5_pipeline/pipeline.py \
  --config-path era5_pipeline/pipeline_config.yml \
  "$@"
