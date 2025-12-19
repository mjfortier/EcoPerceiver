#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=simple_inference.out
#SBATCH --error=simple_inference.error
#SBATCH --job-name=simple_inference
#SBATCH --account=aip-pal

module load python/3.12
source /home/l/luislara/links/scratch/env/ecoperceiver/bin/activate
cd ~/links/scratch/EcoPerceiver/experiments
python simple_inference.py
