#!/bin/bash

# Stage 2: STEGO Clustering
# Unsupervised clustering of DINO features

echo "=========================================="
echo "Starting STEGO Clustering (Stage 2)"
echo "=========================================="

python support/medical_stego/training/train_stego.py \
    --data_dir data/brats_slices/train \
    --dino_checkpoint checkpoints/dino/best.pt \
    --output_dir checkpoints/stego \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4

echo ""
echo "STEGO clustering completed!"
echo "Checkpoint saved to: checkpoints/stego/best.pt"
