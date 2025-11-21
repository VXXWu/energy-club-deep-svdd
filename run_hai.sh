#!/bin/bash

# Activate virtual environment if needed (assuming user has one or we use system python)
# source myenv/bin/activate

# Create log directory
mkdir -p log/hai_test

# Run experiment
# Using hai dataset, hai_mlp network
# data path points to the cloned repo root
./venv/bin/python src/main.py hai hai_mlp log/hai_test data/hai \
    --objective one-class \
    --lr 0.0001 \
    --n_epochs 25 \
    --lr_milestone 20 \
    --batch_size 64 \
    --weight_decay 0.5e-6 \
    --pretrain True \
    --ae_lr 0.0001 \
    --ae_n_epochs 25 \
    --ae_lr_milestone 20 \
    --ae_batch_size 64 \
    --ae_weight_decay 0.5e-6 \
    --normal_class 0 \
    --device mps # Use MPS for Mac GPU acceleration
