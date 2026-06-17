#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.error
#SBATCH --job-name=eval
#SBATCH --account=aip-pal

source $SCRATCH/env/ecoperceiver/bin/activate
cd ~/links/scratch/EcoPerceiver
RUN_PATH="experiments/runs/final_v2_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0"

PYTHONPATH=. python eval/test_sites.py \
    --run-path "$RUN_PATH" \
    --checkpoint-path checkpoint-11.pth \
    --batch-size 2048 \
    --num-workers 8 \
    --data-dir "/home/l/luislara/links/projects/aip-pal/luislara/carbonsense_v2" 

PYTHONPATH=. python eval/export_latex_tables.py \
    --run_folder "$RUN_PATH"
