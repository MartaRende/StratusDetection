#!/bin/bash
#SBATCH --job-name=train_model    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --gres=shard:24          # number of gpu shards to use
#SBATCH --output=logs/%x_%j.out      # standard output
#SBATCH --error=logs/%x_%j.err       # standard error
#SBATCH --nodelist=chacha            # nodes to use
#module purge
apptainer exec --nv --bind /data/datasets/photocast:/data/datasets/photocast /data/datasets/marta.rende/train.sif python3 -u training.py 1 2 3  