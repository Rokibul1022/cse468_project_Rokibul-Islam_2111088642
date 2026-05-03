"""
Stage 1: DINO Pretraining
==========================
Unsupervised feature learning using self-distillation.

Goal: Learn meaningful MRI features without labels
Method: Teacher-student framework where student learns to match teacher
Duration: 13 epochs
Expected Loss: ~0.95

Key Concepts:
- Teacher: Exponential moving average (EMA) of student weights
- Student: Regular model updated with gradients
- No labels needed: Model learns from image structure itself
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from support.medical_stego.models.dino_mri import MRIDinoWrapper
from support.medical_stego.data.brats_dataset import BraTSDataset
from support.medical_stego.losses.losses import DINOLoss
from support.medical_stego.training.utils import save_checkpoint


def train_dino(
    data_dir='data/brats_slices/train',
    output_dir='checkpoints/dino',
    num_epochs=13,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
):
    """
    Train DINO model for unsupervised feature learning.
    
    Args:
        data_dir: Path to training data
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs (default: 13)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-4)
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("STAGE 1: DINO PRETRAINING")
    print("="*60)
    print(f"Data: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset (no masks needed for unsupervised learning)
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
    
    # Initialize models
    student = MRIDinoWrapper().to(device)
    teacher = MRIDinoWrapper().to(device)
    
    # Teacher is EMA of student (no gradients)
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
    
    print("Student and Teacher models initialized")
    print()
    
    # Loss and optimizer
    criterion = DINOLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        student.train()
        teacher.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            student_out = student(images)['patch_features']
            
            with torch.no_grad():
                teacher_out = teacher(images)['patch_features']
            
            # Flatten spatial dimensions for loss
            B, D, H, W = student_out.shape
            student_flat = student_out.view(B, D, -1).mean(dim=2)  # (B, D)
            teacher_flat = teacher_out.view(B, D, -1).mean(dim=2)  # (B, D)
            
            # Compute loss
            loss = criterion(student_flat, teacher_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher with EMA (momentum = 0.996)
            with torch.no_grad():
                for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                    teacher_param.data = 0.996 * teacher_param.data + 0.004 * student_param.data
            
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
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        save_checkpoint(checkpoint, Path(output_dir) / f'epoch_{epoch}.pt')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(checkpoint, Path(output_dir) / 'best.pt')
            print(f"✓ Best model saved (loss: {best_loss:.4f})")
            print()
    
    print("="*60)
    print("DINO PRETRAINING COMPLETED")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Run training
    train_dino(device=device)
