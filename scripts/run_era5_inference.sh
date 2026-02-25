#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load python/3.12

printf "\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/ecoperceiverenv
source $SLURM_TMPDIR/ecoperceiverenv/bin/activate

pip install --upgrade pip --no-index

printf "\nInstalling EcoPerceiver dependencies."
pip install -e . --no-cache-dir --ignore-installed --no-index
pip install scipy --no-index

printf "\nExecuting the inference script."
cd eval/
python3 era5_db_launch.py
python3 test_era5.py
