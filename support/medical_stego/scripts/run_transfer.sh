#!/bin/bash

# Stage 4: Transfer Learning to ISLES
# Domain adaptation from brain tumors to stroke lesions

echo "=========================================="
echo "Starting Transfer Learning (Stage 4)"
echo "=========================================="
echo ""
echo "Domain Adaptation:"
echo "  Source: BraTS (brain tumors, 4 classes)"
echo "  Target: ISLES (stroke lesions, 2 classes)"
echo ""

# Set freeze option (default: true = freeze encoder)
export FREEZE_ENCODER=${FREEZE_ENCODER:-true}

if [ "$FREEZE_ENCODER" = "true" ]; then
    echo "Mode: Frozen encoder (only train decoder)"
else
    echo "Mode: Fine-tune encoder (train all layers)"
fi
echo ""

python support/medical_stego/training/train_transfer.py \
    --train_dir data/isles_slices/train \
    --val_dir data/isles_slices/val \
    --brats_checkpoint checkpoints/finetune/fraction_0.1_best.pt \
    --output_dir checkpoints/transfer \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --freeze_encoder $FREEZE_ENCODER

echo ""
echo "Transfer learning completed!"
echo "Checkpoint saved to: checkpoints/transfer/best.pt"
echo ""
echo "To run with unfrozen encoder:"
echo "  FREEZE_ENCODER=false bash support/medical_stego/scripts/run_transfer.sh"
