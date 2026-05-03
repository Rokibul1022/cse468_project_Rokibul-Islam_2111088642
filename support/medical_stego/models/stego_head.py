import torch
import torch.nn as nn

class STEGOProjectionHead(nn.Module):
    def __init__(self, in_dim=384, proj_dim=128, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, proj_dim, 1)
        )
        
        # Optional cluster probe (for STEGO training)
        self.cluster_probe = None
        
    def forward(self, x):
        # x: [B, 384, 28, 28] (from patch_size=8 DINO)
        projected = self.projection(x)  # [B, proj_dim, 28, 28]
        
        # Already at 28x28, no need to upsample
        # (Decoder will handle upsampling to 224x224)
        
        return projected, None, None  # Return tuple for compatibility
