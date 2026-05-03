#!/bin/bash

# Stage 1: DINO Pretraining
# Unsupervised feature learning from MRI images

echo "=========================================="
echo "Starting DINO Pretraining (Stage 1)"
echo "=========================================="

python support/medical_stego/training/train_dino.py \
    --data_dir data/brats_slices/train \
    --output_dir checkpoints/dino \
    --num_epochs 13 \
    --batch_size 32 \
    --learning_rate 1e-4

echo ""
echo "DINO pretraining completed!"
echo "Checkpoint saved to: checkpoints/dino/best.pt"
