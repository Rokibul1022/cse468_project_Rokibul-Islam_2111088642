"""
3D Decoder for Stroke Lesion Segmentation
==========================================
Converts 2D DINO+STEGO features to 3D predictions.

Strategy:
1. Process each 2D slice with DINO+STEGO (pretrained)
2. Stack features into 3D volume
3. Apply 3D convolutions for volumetric context
4. Output 3D segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder3D(nn.Module):
    """
    3D decoder that processes stacked 2D features.
    
    Args:
        in_channels: Input channels from STEGO (128)
        num_classes: Output classes (2 for ISLES)
        feature_size: Spatial size of 2D features (28x28)
    """
    
    def __init__(self, in_channels=128, num_classes=2, feature_size=28):
        super().__init__()
        self.feature_size = feature_size
        
        # 3D convolutions for volumetric context
        self.conv3d_1 = nn.Conv3d(in_channels, 256, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm3d(256)
        
        self.conv3d_2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn3d_2 = nn.BatchNorm3d(128)
        
        self.conv3d_3 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        
        # Final classification
        self.conv3d_final = nn.Conv3d(64, num_classes, kernel_size=1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv3d_final.weight)
    
    def forward(self, x):
        """
        Args:
            x: Stacked 2D features (B, C, H, W, D)
               where D is number of slices
        
        Returns:
            logits: 3D segmentation (B, num_classes, H_out, W_out, D_out)
        """
        # x: (B, 128, 28, 28, D)
        
        # 3D convolutions
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))  # (B, 256, 28, 28, D)
        x = F.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=False)  # (B, 256, 56, 56, D)
        
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))  # (B, 128, 56, 56, D)
        x = F.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=False)  # (B, 128, 112, 112, D)
        
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))  # (B, 64, 112, 112, D)
        x = F.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=False)  # (B, 64, 224, 224, D)
        
        # Final classification
        logits = self.conv3d_final(x)  # (B, 2, 224, 224, D)
        
        return logits


class Hybrid2D3DModel(nn.Module):
    """
    Hybrid model: 2D feature extraction + 3D segmentation.
    
    Uses pretrained 2D DINO+STEGO for each slice,
    then 3D decoder for volumetric context.
    """
    
    def __init__(self, dino, stego, decoder_3d):
        super().__init__()
        self.dino = dino
        self.stego = stego
        self.decoder_3d = decoder_3d
        
        # Freeze DINO and STEGO
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.stego.parameters():
            param.requires_grad = False
    
    def forward(self, volume):
        """
        Args:
            volume: 3D MRI volume (B, 1, H, W, D)
        
        Returns:
            logits: 3D segmentation (B, 2, H, W, D)
        """
        B, C, H, W, D = volume.shape
        
        # Process each slice with 2D DINO+STEGO
        slice_features = []
        
        for d in range(D):
            # Extract slice
            slice_2d = volume[:, :, :, :, d]  # (B, 1, H, W)
            
            # Resize to 224x224 for DINO
            slice_2d = F.interpolate(slice_2d, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Repeat to 3 channels for DINO
            slice_3ch = slice_2d.repeat(1, 3, 1, 1)  # (B, 3, 224, 224)
            
            # DINO features
            with torch.no_grad():
                dino_out = self.dino(slice_3ch)
                dino_features = dino_out['patch_features']  # (B, 384, 14, 14)
                
                # STEGO projection
                stego_features, _, _ = self.stego(dino_features)  # (B, 128, 28, 28)
            
            slice_features.append(stego_features)
        
        # Stack features into 3D volume
        features_3d = torch.stack(slice_features, dim=-1)  # (B, 128, 28, 28, D)
        
        # 3D decoder
        logits = self.decoder_3d(features_3d)  # (B, 2, 224, 224, D)
        
        # Resize to match input size
        logits = F.interpolate(logits, size=(H, W, D), mode='trilinear', align_corners=False)
        
        return logits
