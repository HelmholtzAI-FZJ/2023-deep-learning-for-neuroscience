#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2303

# Use this on tuesday
##SBATCH --reservation=training-20230229
# And this one on wednesday
##SBATCH --reservation=training-20230301

mkdir -p output
source sc_venv_template/activate.sh


srun python -u loader_timer.py