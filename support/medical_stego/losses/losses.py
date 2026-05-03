"""
Loss Functions
==============
Custom loss functions for brain tumor segmentation.

1. Weighted Cross-Entropy Loss:
   - Addresses class imbalance by weighting rare classes higher
   - Weights: [1.0, 50.0, 15.0, 30.0] for [background, necrotic, edema, enhancing]
   
2. DINO Self-Distillation Loss:
   - Teacher-student framework for unsupervised learning
   - Student learns to match teacher's predictions
   
3. STEGO Contrastive Loss:
   - Pulls similar pixels together in feature space
   - Pushes dissimilar pixels apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class weights to handle imbalance.
    
    Args:
        weights: List of weights for each class [background, necrotic, edema, enhancing]
                 Default: [1.0, 50.0, 15.0, 30.0] based on inverse frequency
    """
    
    def __init__(self, weights=None):
        super().__init__()
        
        if weights is None:
            # Default weights based on BraTS class distribution
            weights = [1.0, 50.0, 15.0, 30.0]
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Weighted CE Loss initialized with weights: {weights}")
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, 4, H, W)
            targets: Ground truth labels (B, H, W) with values 0,1,2,3
        
        Returns:
            loss: Scalar loss value
        """
        # Move weights to same device as logits
        weights = self.weights.to(logits.device)
        
        # Compute cross-entropy with class weights
        loss = F.cross_entropy(logits, targets, weight=weights)
        
        return loss


class DINOLoss(nn.Module):
    """
    DINO self-distillation loss for unsupervised pretraining.
    
    How it works:
    1. Teacher network (EMA of student) produces "soft" targets
    2. Student network learns to match teacher predictions
    3. Temperature controls sharpness of distributions
    """
    
    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: Student predictions (B, D)
            teacher_output: Teacher predictions (B, D)
        
        Returns:
            loss: Cross-entropy between student and teacher distributions
        """
        # Normalize and apply temperature
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        return loss


class STEGOLoss(nn.Module):
    """
    STEGO contrastive loss for unsupervised clustering.
    
    Components:
    1. Contrastive loss: Similar pixels should have similar features
    2. Boundary loss: Preserve edges between different regions
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, positive_pairs, negative_pairs):
        """
        Args:
            features: Projected features (B, D, H, W)
            positive_pairs: Indices of similar pixel pairs
            negative_pairs: Indices of dissimilar pixel pairs
        
        Returns:
            loss: Contrastive loss value
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        B, D, H, W = features.shape
        features_flat = features.view(B, D, -1).permute(0, 2, 1)  # (B, HW, D)
        
        # Cosine similarity
        similarity = torch.bmm(features_flat, features_flat.transpose(1, 2))  # (B, HW, HW)
        similarity = similarity / self.temperature
        
        # Contrastive loss (simplified version)
        # Pull positive pairs together, push negative pairs apart
        pos_loss = -torch.log(torch.sigmoid(similarity[positive_pairs])).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(similarity[negative_pairs])).mean()
        
        loss = pos_loss + neg_loss
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation evaluation.
    
    Dice coefficient measures overlap between prediction and ground truth:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Used for validation, not training (too unstable for training).
    """
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets, class_id=None):
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth (B, H, W)
            class_id: If specified, compute Dice for single class
        
        Returns:
            dice: Dice coefficient (higher is better)
        """
        # Get predictions
        preds = torch.argmax(logits, dim=1)  # (B, H, W)
        
        if class_id is not None:
            # Compute Dice for specific class
            pred_mask = (preds == class_id).float()
            target_mask = (targets == class_id).float()
        else:
            # Compute mean Dice across all classes
            dice_scores = []
            for c in range(logits.shape[1]):
                pred_mask = (preds == c).float()
                target_mask = (targets == c).float()
                
                intersection = (pred_mask * target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()
                
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice)
            
            return torch.stack(dice_scores).mean()
        
        # Compute Dice
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return dice
