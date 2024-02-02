#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --job-name=multi
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yunruili@brandeis.edu
#SBATCH --output=multi-%j.txt
#SBATCH --gres=gpu:V100:2

module swap gnu7/7.3.0 gnu/5.4.0
module load cuda/10.2

python main_GNN_multitask_alignment.py --hidden_channels $1 --num_layers $2 --num_output_layers $3
