#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --job-name=clip_c
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yunruili@brandeis.edu
#SBATCH --output="base_%j.txt"
#SBATCH --gres=gpu:V100:2

module swap gnu7/7.3.0 gnu/5.4.0
module load cuda/10.2

python main_base.py --n_layers $1 --initial_hidden_size $2
