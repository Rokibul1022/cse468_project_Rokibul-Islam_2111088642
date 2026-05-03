"""
Stage 4: Transfer Learning to ISLES
====================================
Domain adaptation from brain tumors (BraTS) to stroke lesions (ISLES).

Goal: Adapt tumor segmentation model to stroke lesion segmentation
Method: Fine-tune decoder on ISLES data using pretrained BraTS features
Duration: 50 epochs
Expected Improvement: Better stroke lesion detection

Key Concepts:
- Transfer learning: Reuse features learned from BraTS
- Domain adaptation: Tumors → Stroke lesions
- Binary segmentation: 2 classes (background, lesion) vs 4 classes (BraTS)
- Smaller decoder: Output 2 classes instead of 4

Why This Works:
1. Both are brain MRI images (similar domain)
2. Both involve abnormal tissue detection
3. DINO+STEGO features are general (work for both tasks)
4. Only need to adapt final decoder layer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from support.medical_stego.models.dino_mri import MRIDinoWrapper
from support.medical_stego.models.stego_head import STEGOProjectionHead
from support.medical_stego.models.full_model import SimpleDecoder
from support.medical_stego.data.isles_dataset import ISLESDataset
from support.medical_stego.losses.losses import WeightedCrossEntropyLoss, DiceLoss
from support.medical_stego.training.utils import load_checkpoint, save_checkpoint


def train_transfer(
    train_dir='data/isles_slices/train',
    val_dir='data/isles_slices/val',
    brats_checkpoint='checkpoints/finetune/fraction_0.1_best.pt',
    output_dir='checkpoints/transfer',
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    freeze_encoder=True,
    device='cuda'
):
    """
    Transfer learning from BraTS to ISLES.
    
    Args:
        train_dir: Path to ISLES training data
        val_dir: Path to ISLES validation data
        brats_checkpoint: Path to trained BraTS model
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs (default: 50)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        freeze_encoder: If True, freeze DINO+STEGO (default: True)
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("STAGE 4: TRANSFER LEARNING TO ISLES")
    print("="*60)
    print(f"Source: BraTS (brain tumors)")
    print(f"Target: ISLES (stroke lesions)")
    print(f"Train data: {train_dir}")
    print(f"Val data: {val_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Freeze encoder: {freeze_encoder}")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ISLES datasets
    train_dataset = ISLESDataset(train_dir, return_mask=True)
    val_dataset = ISLESDataset(val_dir, return_mask=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Check class distribution
    class_dist = train_dataset.get_class_distribution()
    total_pixels = sum(class_dist.values())
    lesion_pct = class_dist[1] / total_pixels * 100
    print(f"Class distribution:")
    print(f"  Background: {class_dist[0]/total_pixels*100:.2f}%")
    print(f"  Lesion: {lesion_pct:.2f}%")
    print()
    
    # Data loaders
    # Class-balanced sampler for handling extreme imbalance
    from support.medical_stego.data.class_balanced_sampler import ClassBalancedSampler
    sampler = ClassBalancedSampler(train_dataset, oversample_factor=20)  # Very aggressive
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use class-balanced sampler
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load pretrained BraTS model
    print("Loading pretrained BraTS model...")
    brats_ckpt = load_checkpoint(brats_checkpoint)
    
    # Load DINO
    dino = MRIDinoWrapper().to(device)
    dino_ckpt = load_checkpoint(brats_ckpt['dino_checkpoint'])
    dino.load_state_dict(dino_ckpt['student_state_dict'])
    
    # Load STEGO
    stego = STEGOProjectionHead().to(device)
    stego.load_state_dict(brats_ckpt['stego_state_dict'], strict=False)
    
    print("✓ DINO and STEGO loaded from BraTS checkpoint")
    
    # Freeze or unfreeze encoder
    if freeze_encoder:
        dino.eval()
        stego.eval()
        for param in dino.parameters():
            param.requires_grad = False
        for param in stego.parameters():
            param.requires_grad = False
        print("✓ DINO and STEGO frozen")
    else:
        dino.train()
        stego.train()
        print("✓ DINO and STEGO unfrozen (will be fine-tuned)")
    
    print()
    
    # Create new decoder for binary segmentation (2 classes)
    decoder = SimpleDecoder(in_ch=128, out_ch=2).to(device)
    
    # Option 1: Initialize from scratch (random weights)
    print("✓ New decoder initialized for binary segmentation (2 classes)")
    
    # Option 2: Transfer weights from BraTS decoder (optional)
    # This can help if stroke lesions are similar to tumor classes
    # Uncomment to use:
    # brats_decoder_state = brats_ckpt['decoder_state_dict']
    # # Copy weights for first 2 classes (background + lesion)
    # decoder.conv3.weight.data[:2] = brats_decoder_state['conv3.weight'][:2]
    # decoder.conv3.bias.data[:2] = brats_decoder_state['conv3.bias'][:2]
    # print("✓ Decoder initialized with BraTS weights (first 2 classes)")
    
    print()
    
    # Loss functions
    # Adjust weights based on ISLES class distribution
    # Use EXTREME weight for tiny lesion class (0.32%)
    lesion_weight = 2000.0  # Very aggressive for tiny lesions
    train_criterion = WeightedCrossEntropyLoss(weights=[1.0, lesion_weight])
    dice_metric = DiceLoss()
    
    print(f"Loss weights: [1.0, {lesion_weight:.1f}] (background, lesion)")
    print()
    
    # Optimizer
    if freeze_encoder:
        # Only optimize decoder
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    else:
        # Optimize all parameters (encoder + decoder)
        optimizer = torch.optim.Adam(
            list(dino.parameters()) + list(stego.parameters()) + list(decoder.parameters()),
            lr=learning_rate
        )
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        decoder.train()
        if not freeze_encoder:
            dino.train()
            stego.train()
        
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            if freeze_encoder:
                with torch.no_grad():
                    dino_features = dino(images)['patch_features']
                    stego_features, _, _ = stego(dino_features)
            else:
                dino_features = dino(images)['patch_features']
                stego_features, _, _ = stego(dino_features)
            
            # Decoder forward
            logits = decoder(stego_features)
            
            # Compute loss
            loss = train_criterion(logits, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase (every 5 epochs)
        if epoch % 5 == 0:
            decoder.eval()
            if not freeze_encoder:
                dino.eval()
                stego.eval()
            
            val_dice_scores = {0: [], 1: []}
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    dino_features = dino(images)['patch_features']
                    stego_features, _, _ = stego(dino_features)
                    logits = decoder(stego_features)
                    
                    # Apply AGGRESSIVE temperature scaling for tiny lesions
                    temperature = torch.tensor([1.0, 0.1]).to(device).view(1, 2, 1, 1)
                    logits = logits / temperature
                    
                    # Compute Dice for each class
                    for class_id in range(2):
                        dice = dice_metric(logits, masks, class_id=class_id)
                        val_dice_scores[class_id].append(dice.item())
            
            # Average Dice scores
            avg_dice = {c: sum(scores)/len(scores) for c, scores in val_dice_scores.items()}
            lesion_dice = avg_dice[1]
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Dice Scores:")
            print(f"    Background: {avg_dice[0]*100:.1f}%")
            print(f"    Lesion:     {avg_dice[1]*100:.1f}%")
            print()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'stego_state_dict': stego.state_dict(),
                'dino_state_dict': dino.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'dice_scores': avg_dice,
                'lesion_dice': lesion_dice,
                'brats_checkpoint': brats_checkpoint,
                'freeze_encoder': freeze_encoder
            }
            
            save_checkpoint(checkpoint, output_dir / f'epoch_{epoch}.pt')
            
            # Save best model
            if lesion_dice > best_dice:
                best_dice = lesion_dice
                save_checkpoint(checkpoint, output_dir / 'best.pt')
                print(f"✓ Best model saved (Lesion Dice: {best_dice*100:.1f}%)")
                print()
        else:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print()
    
    print("="*60)
    print("TRANSFER LEARNING COMPLETED")
    print(f"Best Lesion Dice: {best_dice*100:.1f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)
    print()
    print("Transfer Learning Summary:")
    print(f"  Source domain: BraTS (brain tumors, 4 classes)")
    print(f"  Target domain: ISLES (stroke lesions, 2 classes)")
    print(f"  Method: {'Frozen encoder' if freeze_encoder else 'Fine-tuned encoder'}")
    print(f"  Result: {best_dice*100:.1f}% lesion Dice")
    print()
    print("Generating results visualizations...")
    
    # Auto-generate results
    try:
        from support.medical_stego.training.generate_stage4_results import generate_stage4_results
        generate_stage4_results(
            test_dir=val_dir,
            checkpoint=str(output_dir / 'best.pt'),
            output_dir='results/stage4_transfer',
            num_samples=6,
            device=device
        )
        print("✓ Results saved to results/stage4_transfer/")
    except Exception as e:
        print(f"⚠ Could not auto-generate results: {e}")
        print("  Run manually: python support/medical_stego/training/generate_stage4_results.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Transfer learning to ISLES')
    parser.add_argument('--train_dir', type=str, default='data/isles_slices/train')
    parser.add_argument('--val_dir', type=str, default='data/isles_slices/val')
    parser.add_argument('--brats_checkpoint', type=str, default='checkpoints/finetune/fraction_0.1_best.pt')
    parser.add_argument('--output_dir', type=str, default='checkpoints/transfer')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Get freeze option from environment variable
    import os
    freeze_encoder = os.environ.get('FREEZE_ENCODER', 'true').lower() == 'true'
    
    train_transfer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        brats_checkpoint=args.brats_checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_encoder=freeze_encoder,
        device=device
    )
