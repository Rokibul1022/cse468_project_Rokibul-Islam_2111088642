import torch
import torch.nn as nn
import timm

class MRIDinoWrapper(nn.Module):
    def __init__(self, arch='vit_small', patch_size=8, img_size=224, load_pretrained=False):
        super().__init__()
        self.embed_dim = 384  # vit_small embedding dimension
        
        # Create Vision Transformer backbone using timm
        # Checkpoint uses patch_size=8, which gives 28x28=784 patches + 1 CLS = 785 tokens
        self.backbone = timm.create_model(
            f'vit_small_patch{patch_size}_{img_size}',
            pretrained=load_pretrained,
            num_classes=0,  # No classification head
            global_pool=''  # No global pooling
        )
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        
        # Forward through ViT backbone
        features = self.backbone.forward_features(x)  # [B, N+1, D] where N+1 includes CLS token
        
        # Remove CLS token (first token)
        patch_features = features[:, 1:, :]  # [B, N, D] where N=784 for 224x224 with patch_size=8
        
        # Reshape to spatial format
        B, N, D = patch_features.shape
        H = W = int(N ** 0.5)  # Should be 28x28 for patch_size=8
        
        # Handle non-perfect squares
        if H * W != N:
            # Find best factorization
            for h in range(int(N**0.5), 0, -1):
                if N % h == 0:
                    H = h
                    W = N // h
                    break
        
        patch_features = patch_features.transpose(1, 2).reshape(B, D, H, W)
        
        return {'patch_features': patch_features}
