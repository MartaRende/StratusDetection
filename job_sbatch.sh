#!/bin/bash
#SBATCH --job-name=train_model    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs/%x_%j.out      # standard output
#SBATCH --error=logs/%x_%j.err       # standard error
#module purge
apptainer exec --nv ./train.sif python3 training.py 1