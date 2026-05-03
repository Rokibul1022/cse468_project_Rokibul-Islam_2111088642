#!/bin/bash

# Stage 3: Fine-tuning with Limited Labels
# Supervised training with only 10% labeled data

echo "=========================================="
echo "Starting Fine-tuning (Stage 3)"
echo "=========================================="

# Set label fraction (default: 0.1 = 10%)
export LABEL_FRACTION=${LABEL_FRACTION:-0.1}

echo "Using ${LABEL_FRACTION} ($(echo "$LABEL_FRACTION * 100" | bc)%) of labeled data"
echo ""

python support/medical_stego/training/train_finetune.py \
    --train_dir data/brats_slices/train \
    --val_dir data/brats_slices/val \
    --dino_checkpoint checkpoints/dino/best.pt \
    --stego_checkpoint checkpoints/stego/best.pt \
    --output_dir checkpoints/finetune \
    --label_fraction $LABEL_FRACTION \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4

echo ""
echo "Fine-tuning completed!"
echo "Checkpoint saved to: checkpoints/finetune/fraction_${LABEL_FRACTION}_best.pt"
