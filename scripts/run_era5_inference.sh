#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

printf "\\nLoading required modules.\\n"
module load proj/9.2.0 python/3.12
REPO_DIR="$HOME/scratch/CarbonCast/EcoPerceiver"

printf "\\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/ccenv
source $SLURM_TMPDIR/ccenv/bin/activate

pip install --upgrade pip --no-index

printf "\\nInstalling EcoPerceiver dependencies."
cd "$REPO_DIR"
pip install -e . --no-cache-dir --no-index
pip install scipy --no-index
pip install h5py h5netcdf --no-index

printf "\\nExecuting the inference script."
cd eval/
python3 era5_db_launch.py
python3 test_era5.py