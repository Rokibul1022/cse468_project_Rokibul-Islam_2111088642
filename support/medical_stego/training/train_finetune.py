"""
Stage 3: Fine-tuning with 10% Labels
=====================================
Supervised training with limited labeled data.

Goal: Learn to segment tumors using only 10% of labeled data
Method: Fine-tune decoder on top of frozen DINO+STEGO features
Duration: 100 epochs
Expected Loss: ~0.44

Key Techniques:
1. Class-balanced sampling: Oversample rare tumor classes
2. Weighted loss: Higher weight for rare classes [1, 50, 15, 30]
3. BatchNorm in decoder: Stabilizes training
4. Temperature scaling: Reduces background bias at inference

Results:
- Background Dice: 96.8%
- Necrotic Dice: 21.6%
- Edema Dice: 3.1%
- Enhancing Dice: 18.8%
- Mean Tumor Dice: 14.5%
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
from support.medical_stego.data.brats_dataset import BraTSDataset
from support.medical_stego.data.class_balanced_sampler import ClassBalancedSampler
from support.medical_stego.losses.losses import WeightedCrossEntropyLoss, DiceLoss
from support.medical_stego.training.utils import load_checkpoint, save_checkpoint


def train_finetune(
    train_dir='data/brats_slices/train',
    val_dir='data/brats_slices/val',
    dino_checkpoint='checkpoints/dino/best.pt',
    stego_checkpoint='checkpoints/stego/best.pt',
    output_dir='checkpoints/finetune',
    label_fraction=0.1,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
):
    """
    Fine-tune segmentation decoder with limited labels.
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        dino_checkpoint: Path to pretrained DINO
        stego_checkpoint: Path to pretrained STEGO
        output_dir: Where to save checkpoints
        label_fraction: Fraction of labels to use (default: 0.1 = 10%)
        num_epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("STAGE 3: FINE-TUNING WITH LIMITED LABELS")
    print("="*60)
    print(f"Train data: {train_dir}")
    print(f"Val data: {val_dir}")
    print(f"Label fraction: {label_fraction*100}%")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = BraTSDataset(train_dir, return_mask=True)
    val_dataset = BraTSDataset(val_dir, return_mask=True)
    
    # Use only fraction of training data
    num_train = int(len(train_dataset) * label_fraction)
    train_dataset.image_files = train_dataset.image_files[:num_train]
    
    print(f"Training samples: {len(train_dataset)} ({label_fraction*100}% of full dataset)")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Class-balanced sampler for handling imbalance
    sampler = ClassBalancedSampler(train_dataset, oversample_factor=5)
    
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
    
    # Load pretrained models (frozen)
    dino = MRIDinoWrapper().to(device)
    dino_ckpt = load_checkpoint(dino_checkpoint)
    dino.load_state_dict(dino_ckpt['student_state_dict'])
    dino.eval()
    
    stego = STEGOProjectionHead().to(device)
    stego_ckpt = load_checkpoint(stego_checkpoint)
    stego.load_state_dict(stego_ckpt['stego_state_dict'])
    stego.eval()
    
    # Freeze DINO and STEGO
    for param in dino.parameters():
        param.requires_grad = False
    for param in stego.parameters():
        param.requires_grad = False
    
    print("✓ DINO and STEGO loaded and frozen")
    
    # Initialize decoder (trainable)
    decoder = SimpleDecoder(in_ch=128, out_ch=4).to(device)
    print("✓ Decoder initialized")
    print()
    
    # Loss functions
    # Weights based on inverse class frequency
    train_criterion = WeightedCrossEntropyLoss(weights=[1.0, 50.0, 15.0, 30.0])
    dice_metric = DiceLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        decoder.train()
        
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (DINO and STEGO frozen)
            with torch.no_grad():
                dino_features = dino(images)['patch_features']
                stego_features, _, _ = stego(dino_features)
            
            # Decoder forward (trainable)
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
            
            val_dice_scores = {0: [], 1: [], 2: [], 3: []}
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    dino_features = dino(images)['patch_features']
                    stego_features, _, _ = stego(dino_features)
                    logits = decoder(stego_features)
                    
                    # Compute Dice for each class
                    for class_id in range(4):
                        dice = dice_metric(logits, masks, class_id=class_id)
                        val_dice_scores[class_id].append(dice.item())
            
            # Average Dice scores
            avg_dice = {c: sum(scores)/len(scores) for c, scores in val_dice_scores.items()}
            mean_tumor_dice = (avg_dice[1] + avg_dice[2] + avg_dice[3]) / 3
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Dice Scores:")
            print(f"    Background: {avg_dice[0]*100:.1f}%")
            print(f"    Necrotic:   {avg_dice[1]*100:.1f}%")
            print(f"    Edema:      {avg_dice[2]*100:.1f}%")
            print(f"    Enhancing:  {avg_dice[3]*100:.1f}%")
            print(f"    Mean Tumor: {mean_tumor_dice*100:.1f}%")
            print()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'stego_state_dict': stego.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'dice_scores': avg_dice,
                'mean_tumor_dice': mean_tumor_dice,
                'dino_checkpoint': dino_checkpoint,
                'stego_checkpoint': stego_checkpoint,
                'label_fraction': label_fraction
            }
            
            save_checkpoint(checkpoint, output_dir / f'epoch_{epoch}.pt')
            
            # Save best model
            if mean_tumor_dice > best_dice:
                best_dice = mean_tumor_dice
                save_checkpoint(checkpoint, output_dir / f'fraction_{label_fraction}_best.pt')
                print(f"✓ Best model saved (Mean Tumor Dice: {best_dice*100:.1f}%)")
                print()
        else:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print()
    
    print("="*60)
    print("FINE-TUNING COMPLETED")
    print(f"Best Mean Tumor Dice: {best_dice*100:.1f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Get label fraction from environment variable or use default
    import os
    label_fraction = float(os.environ.get('LABEL_FRACTION', '0.1'))
    
    train_finetune(label_fraction=label_fraction, device=device)
