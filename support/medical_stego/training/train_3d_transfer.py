"""
Stage 4: 3D Transfer Learning
==============================
Train 3D decoder for stroke lesion segmentation.

Key improvements over 2D:
- Processes full 3D volumes (not single slices)
- Captures volumetric context for tiny lesions
- Should achieve 30-50% lesion Dice
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from support.medical_stego.models.dino_mri import MRIDinoWrapper
from support.medical_stego.models.stego_head import STEGOProjectionHead
from support.medical_stego.models.decoder_3d import Decoder3D, Hybrid2D3DModel
from support.medical_stego.data.isles_3d_dataset import ISLES3DDataset
from support.medical_stego.losses.losses import WeightedCrossEntropyLoss, DiceLoss
from support.medical_stego.training.utils import load_checkpoint, save_checkpoint


def train_3d_transfer(
    data_dir='data/ISLES-2022/isles-2022',
    brats_checkpoint='checkpoints/finetune/fraction_0.1_best.pt',
    output_dir='checkpoints/transfer_3d',
    num_epochs=50,
    batch_size=2,  # Small batch for 3D (memory intensive)
    learning_rate=1e-4,
    device='cuda'
):
    """
    Train 3D transfer learning model.
    
    Args:
        data_dir: Path to ISLES data
        brats_checkpoint: Path to BraTS model
        output_dir: Where to save checkpoints
        num_epochs: Training epochs
        batch_size: Batch size (small for 3D)
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("STAGE 4: 3D TRANSFER LEARNING")
    print("="*60)
    print(f"Method: 2D features + 3D decoder")
    print(f"Patch size: 128x128x16")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    data_dir = Path(data_dir)
    derivatives_dir = data_dir / 'derivatives'
    
    # Create full datasets first to get valid cases
    full_train_dataset = ISLES3DDataset(
        data_dir, derivatives_dir,
        patch_size=(128, 128, 16),
        patches_per_volume=4
    )
    
    full_val_dataset = ISLES3DDataset(
        data_dir, derivatives_dir,
        patch_size=(128, 128, 16),
        patches_per_volume=2
    )
    
    # Split valid cases into train/val
    valid_cases = full_train_dataset.case_dirs
    num_train = int(len(valid_cases) * 0.8)
    train_cases = valid_cases[:num_train]
    val_cases = valid_cases[num_train:]
    
    # Create train/val datasets with filtered cases
    train_dataset = ISLES3DDataset(
        data_dir, derivatives_dir,
        patch_size=(128, 128, 16),
        patches_per_volume=4,
        case_dirs=train_cases
    )
    
    val_dataset = ISLES3DDataset(
        data_dir, derivatives_dir,
        patch_size=(128, 128, 16),
        patches_per_volume=2,
        case_dirs=val_cases
    )
    
    print(f"Train volumes: {len(train_cases)} ({len(train_dataset)} patches)")
    print(f"Val volumes: {len(val_cases)} ({len(val_dataset)} patches)")
    print()
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    # Load pretrained BraTS models
    print("Loading pretrained BraTS models...")
    brats_ckpt = load_checkpoint(brats_checkpoint)
    
    dino = MRIDinoWrapper().to(device)
    dino_ckpt = load_checkpoint(brats_ckpt['dino_checkpoint'])
    dino.load_state_dict(dino_ckpt['student_state_dict'])
    dino.eval()
    
    stego = STEGOProjectionHead().to(device)
    stego.load_state_dict(brats_ckpt['stego_state_dict'], strict=False)
    stego.eval()
    
    print("✓ DINO and STEGO loaded (frozen)")
    
    # Create 3D decoder
    decoder_3d = Decoder3D(in_channels=128, num_classes=2).to(device)
    print("✓ 3D Decoder initialized")
    print()
    
    # Create hybrid model
    model = Hybrid2D3DModel(dino, stego, decoder_3d)
    
    # Loss and optimizer
    criterion = WeightedCrossEntropyLoss(weights=[1.0, 100.0])  # High weight for lesion
    dice_metric = DiceLoss()
    optimizer = torch.optim.Adam(decoder_3d.parameters(), lr=learning_rate)
    
    print(f"Loss weights: [1.0, 100.0] (background, lesion)")
    print()
    
    # Check for existing checkpoint to resume
    start_epoch = 1
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice_bg': [], 'val_dice_lesion': [], 'epochs': []}
    resume_path = output_dir / 'latest.pt'
    
    if resume_path.exists():
        print(f"Found checkpoint: {resume_path}")
        resume_ckpt = load_checkpoint(resume_path)
        decoder_3d.load_state_dict(resume_ckpt['decoder_3d_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        start_epoch = resume_ckpt['epoch'] + 1
        best_dice = resume_ckpt.get('best_dice', 0.0)
        history = resume_ckpt.get('history', history)
        print(f"✓ Resumed from epoch {resume_ckpt['epoch']}")
        print(f"✓ Best Dice so far: {best_dice*100:.1f}%")
        print()
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        # Training
        model.train()
        decoder_3d.train()
        
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (volumes, masks) in enumerate(train_loader):
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(volumes)  # (B, 2, H, W, D)
            
            # Reshape for loss computation
            B, C, H, W, D = logits.shape
            logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*H*W*D, 2)
            masks_flat = masks.reshape(-1)  # (B*H*W*D,)
            
            # Compute loss
            loss = criterion(logits_flat.unsqueeze(2).unsqueeze(3), 
                           masks_flat.unsqueeze(1).unsqueeze(2))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_3d.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['epochs'].append(epoch)
        
        # Save checkpoint after every epoch for safety
        checkpoint = {
            'epoch': epoch,
            'decoder_3d_state_dict': decoder_3d.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'best_dice': best_dice,
            'history': history,
            'brats_checkpoint': brats_checkpoint
        }
        save_checkpoint(checkpoint, output_dir / 'latest.pt')
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            decoder_3d.eval()
            
            val_dice_scores = {0: [], 1: []}
            
            with torch.no_grad():
                for volumes, masks in val_loader:
                    volumes = volumes.to(device)
                    masks = masks.to(device)
                    
                    # Forward
                    logits = model(volumes)
                    
                    # Compute Dice for each class
                    B, C, H, W, D = logits.shape
                    for d in range(D):
                        logits_slice = logits[:, :, :, :, d]
                        masks_slice = masks[:, :, :, d]
                        
                        for class_id in range(2):
                            dice = dice_metric(logits_slice, masks_slice, class_id=class_id)
                            val_dice_scores[class_id].append(dice.item())
            
            # Average Dice
            avg_dice = {c: sum(scores)/len(scores) if scores else 0.0 
                       for c, scores in val_dice_scores.items()}
            lesion_dice = avg_dice[1]
            
            # Update history with validation metrics
            history['val_dice_bg'].append(avg_dice[0])
            history['val_dice_lesion'].append(lesion_dice)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Dice Scores:")
            print(f"    Background: {avg_dice[0]*100:.1f}%")
            print(f"    Lesion:     {avg_dice[1]*100:.1f}%")
            print()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'decoder_3d_state_dict': decoder_3d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'dice_scores': avg_dice,
                'lesion_dice': lesion_dice,
                'best_dice': best_dice,
                'history': history,
                'brats_checkpoint': brats_checkpoint
            }
            
            save_checkpoint(checkpoint, output_dir / f'epoch_{epoch}.pt')
            save_checkpoint(checkpoint, output_dir / 'latest.pt')  # For resuming
            
            # Save best
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
    print("3D TRANSFER LEARNING COMPLETED")
    print(f"Best Lesion Dice: {best_dice*100:.1f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)
    print()
    
    # Auto-generate results
    print("Generating results visualizations...")
    try:
        from support.medical_stego.training.generate_stage4_results import generate_stage4_results
        generate_stage4_results(
            checkpoint_path=str(output_dir / 'best.pt'),
            output_dir='results/stage4_3d_transfer'
        )
        print("✓ Results saved to: results/stage4_3d_transfer")
    except Exception as e:
        print(f"Warning: Could not generate results: {e}")
        print("You can manually run: python support/medical_stego/training/generate_stage4_results.py")
    
    print("="*60)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    train_3d_transfer(device=device)
