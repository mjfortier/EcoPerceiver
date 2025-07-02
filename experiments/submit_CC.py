import time
import os
import argparse
import sys
import re
import subprocess
import yaml
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./config.yml')
parser.add_argument("--n_nodes", type=int, default=1)
parser.add_argument("--gpus_per_node", type=int, default=4)
parser.add_argument("--hours", type=int, default=8)
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--one_shot", action='store_true', default=True)
parser.add_argument("--prefix", default='')
parser.add_argument("--main", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

if not os.path.exists('runs'):
    print('needs a runs folder (or symlink) in this directory')
    sys.exit()

if not os.path.exists('data'):
    print('neds a data folder (or symlink) in this directory')
    sys.exit()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

run_descriptor = f"{config['optimizer']['lr']}_{'ws' if config['model']['weight_sharing'] else 'nws'}_l{config['model']['latent_space_dim']}_f{config['model']['num_frequencies']}_e{config['model']['input_embedding_dim']}_c{config['model']['context_length']}_o{config['model']['obs_dropout']}_{config['model']['layers']}"
run_name = f"{args.prefix}_{run_descriptor}_CC"
job_name = f"{args.prefix}_{args.seed}_{run_descriptor}_CC"
run_dir = os.path.join(os.path.realpath('runs'), run_name)
seed_dir = os.path.join(os.path.join(run_dir, f'seed_{args.seed}'))

config_loc = os.path.join(run_dir, 'config.yml')

data_dir = os.path.realpath('data')
tensorboard_dir = os.path.join(os.path.realpath('tensorboard'), run_name)

job_script=f"""#!/bin/bash
#SBATCH --nodes={args.n_nodes}
#SBATCH --ntasks-per-node={args.gpus_per_node}
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-{args.hours}:00:00
#SBATCH --output={seed_dir}/%N-%j.out
#SBATCH --error={seed_dir}/%N-%j.error
#SBATCH --job-name={job_name}

module load python
source /home/mfortier/env/scratch/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$SLURMD_NODENAME


srun python /home/mfortier/EcoPerceiver/experiments/run_experiment.py \\
  --run_dir {seed_dir} \\
  --config {config_loc} \\
  --data_dir {data_dir} \\
  --tensorboard_dir {tensorboard_dir} \\
  --auto_resume \\
  --seed {args.seed}

"""

if args.dry_run:
    print(job_script)
    sys.exit()

os.makedirs(run_dir, exist_ok=True)
os.makedirs(seed_dir, exist_ok=True)

job_script_path = os.path.join(seed_dir, 'submit.sh')
with open(job_script_path,'w') as f:
    f.write(job_script)
shutil.copyfile(args.config, config_loc)

os.system(f'sbatch {job_script_path}')
