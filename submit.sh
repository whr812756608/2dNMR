#!/bin/bash -l
#SBATCH --job-name=clip
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:RTX2:1

conda activate clip
python main_GNN_multitask_alignment_HX.py \
    --batch_size $1 \
    --n_epoch $2 \
    --lr $3 \
    --type $4 \
    --hidden_channels $5 \
    --num_layers $6 \
    --num_output_layers $7 \
    --agg_method $8 \
    --c_out_hidden 128 64 \
    --h_out_hidden 128 64

