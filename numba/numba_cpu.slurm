#!/bin/bash
#SBATCH --job-name=numba         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=bootcamp2  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2023.9
conda activate numba-env

python numba_cpu.py
