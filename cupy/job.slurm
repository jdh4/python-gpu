#!/bin/bash
#SBATCH --job-name=cupy-job      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory (RAM) per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=gpu80       # choose gpu80, a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2024.6
conda activate /scratch/network/jdh4/.gpu_workshop/envs/cupy-env

python matmul_numpy.py
python matmul_cupy.py
