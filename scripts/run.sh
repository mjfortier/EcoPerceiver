module load python/3.12
python -m venv $SCRATCH/env/ecoperceiver
source $SCRATCH/env/ecoperceiver/bin/activate

module load StdEnv/2023 gcc/12.3
module load hdf5/1.14.5 netcdf/4.9.2

ssh tc11101
cd ~/links/scratch/EcoPerceiver/experiments

tensorboard --logdir tensorboard/

python submit_CC.py --prefix final_v3 --hours 24 --n_nodes 4
