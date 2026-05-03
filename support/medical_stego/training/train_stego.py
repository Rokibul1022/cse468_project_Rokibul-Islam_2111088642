"""
Stage 2: STEGO Clustering
==========================
Unsupervised clustering of DINO features into semantic regions.

Goal: Group similar pixels together without labels
Method: Contrastive learning on DINO features
Duration: 20 epochs
Expected Loss: ~0.012

Key Concepts:
- Uses frozen DINO features from Stage 1
- Learns projection head to cluster features
- Positive pairs: Nearby pixels (likely same tissue)
- Negative pairs: Distant pixels (likely different tissue)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from support.medical_stego.models.dino_mri import MRIDinoWrapper
from support.medical_stego.models.stego_head import STEGOProjectionHead
from support.medical_stego.data.brats_dataset import BraTSDataset
from support.medical_stego.losses.losses import STEGOLoss
from support.medical_stego.training.utils import load_checkpoint, save_checkpoint


def train_stego(
    data_dir='data/brats_slices/train',
    dino_checkpoint='checkpoints/dino/best.pt',
    output_dir='checkpoints/stego',
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
):
    """
    Train STEGO projection head for unsupervised clustering.
    
    Args:
        data_dir: Path to training data
        dino_checkpoint: Path to pretrained DINO model
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs (default: 20)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("STAGE 2: STEGO CLUSTERING")
    print("="*60)
    print(f"Data: {data_dir}")
    print(f"DINO checkpoint: {dino_checkpoint}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = BraTSDataset(data_dir, return_mask=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Loaded {len(dataset)} training samples")
    print()
    
    # Load pretrained DINO (frozen)
    dino = MRIDinoWrapper().to(device)
    dino_ckpt = load_checkpoint(dino_checkpoint)
    dino.load_state_dict(dino_ckpt['student_state_dict'])
    dino.eval()
    
    # Freeze DINO
    for param in dino.parameters():
        param.requires_grad = False
    
    print("✓ DINO loaded and frozen")
    
    # Initialize STEGO projection head (trainable)
    stego = STEGOProjectionHead().to(device)
    print("✓ STEGO projection head initialized")
    print()
    
    # Loss and optimizer
    criterion = STEGOLoss()
    optimizer = torch.optim.Adam(stego.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        stego.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            B, C, H, W = images.shape
            
            # Extract DINO features (frozen)
            with torch.no_grad():
                dino_features = dino(images)['patch_features']  # (B, 384, 14, 14)
            
            # Project features with STEGO
            projected, _, _ = stego(dino_features)  # (B, 128, 28, 28)
            
            # Generate positive/negative pairs
            # Positive: Nearby pixels (within 3x3 neighborhood)
            # Negative: Distant pixels (>10 pixels apart)
            positive_pairs, negative_pairs = generate_pairs(projected)
            
            # Compute contrastive loss
            loss = criterion(projected, positive_pairs, negative_pairs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'stego_state_dict': stego.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'dino_checkpoint': dino_checkpoint
        }
        
        save_checkpoint(checkpoint, Path(output_dir) / f'epoch_{epoch}.pt')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(checkpoint, Path(output_dir) / 'best.pt')
            print(f"✓ Best model saved (loss: {best_loss:.4f})")
            print()
    
    print("="*60)
    print("STEGO CLUSTERING COMPLETED")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)


def generate_pairs(features):
    """
    Generate positive and negative pixel pairs for contrastive learning.
    
    Positive pairs: Spatially close pixels (likely same tissue type)
    Negative pairs: Spatially distant pixels (likely different tissue)
    
    Args:
        features: Feature map (B, D, H, W)
    
    Returns:
        positive_pairs: Indices of positive pairs
        negative_pairs: Indices of negative pairs
    """
    B, D, H, W = features.shape
    
    # Simplified version: sample random pairs
    # In practice, use spatial proximity
    num_pairs = 1000
    
    # Positive pairs: nearby pixels
    pos_idx1 = torch.randint(0, H*W, (B, num_pairs))
    pos_idx2 = pos_idx1 + torch.randint(-3, 4, (B, num_pairs))  # Within 3 pixels
    pos_idx2 = torch.clamp(pos_idx2, 0, H*W-1)
    
    # Negative pairs: distant pixels
    neg_idx1 = torch.randint(0, H*W, (B, num_pairs))
    neg_idx2 = torch.randint(0, H*W, (B, num_pairs))
    
    positive_pairs = (torch.arange(B).unsqueeze(1), pos_idx1, pos_idx2)
    negative_pairs = (torch.arange(B).unsqueeze(1), neg_idx1, neg_idx2)
    
    return positive_pairs, negative_pairs


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    train_stego(device=device)
