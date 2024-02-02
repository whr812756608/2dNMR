#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --job-name=gnn
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yunruili@brandeis.edu
#SBATCH --output="sp_%j.txt"
#SBATCH --gres=gpu:TitanXP:2

module swap gnu7/7.3.0 gnu/5.4.0
module load cuda/10.2

python main_spherenet.py --batch_size $1 --lr $2 
