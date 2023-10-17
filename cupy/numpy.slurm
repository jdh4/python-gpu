#!/bin/bash
#SBATCH --job-name=numpy-job     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory (RAM) per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=bootcamp2  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2023.9

python matmul_numpy.py
