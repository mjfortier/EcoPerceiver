#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load StdEnv/2023 openmpi/4.1.5 netcdf-mpi/4.9.2 mpi4py/4.0.3 proj/9.2.0 python/3.12

printf "\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/csdpenv
source $SLURM_TMPDIR/csdpenv/bin/activate

pip install --upgrade pip --no-index

printf "\nInstalling EcoPerceiver dependencies."
pip install -e . --no-cache-dir --ignore-installed --no-index
pip install scipy --no-index

export LD_PRELOAD=$EBROOTOPENMPI/lib/libmpi.so

printf "\nExecuting the inference script."
cd eval/
python3 era5_db_launch.py
python3 test_era5.py
