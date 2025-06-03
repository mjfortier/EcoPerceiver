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
parser.add_argument("--max_restarts", type=int, default=3)
parser.add_argument("--sleep_time", type=float, default=1.0) #sleep time in hours before submitting a new job
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--one_shot", action='store_true', default=True)
parser.add_argument("--cluster", default='CC')
args = parser.parse_args()

if not os.path.exists('runs'):
    print('needs a runs folder (or symlink) in this directory')
    sys.exit()

if not os.path.exists('data'):
    print('neds a data folder (or symlink) in this directory')
    sys.exit()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

run_name = config['run']['run_name']
run_dir = os.path.join(os.path.realpath('runs'), run_name)

data_dir = os.path.realpath('data')

if args.cluster == 'CC':
    env_script = """
module load python
source /home/mfortier/env/scratch/bin/activate
"""
elif args.cluster == 'MILA':
    env_script = """
source ~/.bashrc
conda activate scratch
"""
else:
    print('Invalid cluster')
    sys.exit()

job_script=f"""#!/bin/bash
#SBATCH --nodes={args.n_nodes}
#SBATCH --gpus-per-node={args.gpus_per_node}
#SBATCH --tasks-per-node={args.gpus_per_node}
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=0-{args.hours}:00:00
#SBATCH --output={run_dir}/%N-%j.out
#SBATCH --error={run_dir}/%N-%j.error
#SBATCH --job-name={run_name}

{env_script}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(hostname)

srun python run_experiment.py \\
  --run_dir {run_dir} \\
  --data_dir {data_dir} \\
  --world_size {args.n_nodes * args.gpus_per_node} \\
  --gpus_per_node {args.gpus_per_node} \\
  --dist_url tcp://$MASTER_ADDR:$MASTER_PORT
"""

if args.dry_run:
    print(job_script)
    sys.exit()

os.makedirs(run_dir, exist_ok=True)

job_script_path = os.path.join(run_dir, 'submit.sh')
with open(job_script_path,'w') as f:
    f.write(job_script)
shutil.copyfile(args.config, os.path.join(run_dir, 'config.yml'))
if args.one_shot:
    os.system(f'sbatch {job_script_path}')
    sys.exit()

# Continuous process
def find_highest_checkpoint_number(dir):
    max_num = 0
    pattern = re.compile(r"checkpoint-(\d+)\.pth")  # Regex to match and capture the numerical part of the filename
    for filename in os.listdir(dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num

if find_highest_checkpoint_number(run_dir) >= args.num_epochs:
    print('Already finished final epoch. Aborting...')
    sys.exit(1)


def shell_output(command):
    op_raw = subprocess.check_output(command, shell=True)
    return op_raw.decode('ascii')

seconds_to_sleep = args.sleep_time * 60 * 60



for i in range(args.max_restarts):

    job_submit_output = shell_output(f'sbatch {job_script_path}')
    job_id = int(job_submit_output.split(' ')[-1].strip('\n'))
    print(f'Started job {job_id}')
    time.sleep(10)

    while job_id in shell_output('sq'):
        time.sleep(seconds_to_sleep)

