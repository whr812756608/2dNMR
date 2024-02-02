#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --job-name=gen_graph
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yunruili@brandeis.edu
#SBATCH --output=gen_graph.txt
#SBATCH --gres=gpu:V100:1

module swap gnu7/7.3.0 gnu/5.4.0
module load cuda/10.2

python gen_alignment_3dgraph_new.py
