import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoder(nn.Module):
    def __init__(self, in_ch=128, out_ch=4):
        super().__init__()
        # Match checkpoint structure: 128 -> 128 -> 64 -> 32 -> out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),  # net.0
            nn.BatchNorm2d(128),                   # net.1
            nn.ReLU(),                             # net.2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # net.3
            nn.Conv2d(128, 64, 3, padding=1),     # net.4
            nn.BatchNorm2d(64),                    # net.5
            nn.ReLU(),                             # net.6
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # net.7
            nn.Conv2d(64, 32, 3, padding=1),      # net.8
            nn.BatchNorm2d(32),                    # net.9
            nn.ReLU(),                             # net.10
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # net.11
            nn.Conv2d(32, out_ch, 1)              # net.12
        )
        
        # Xavier initialization for final layer
        nn.init.xavier_uniform_(self.net[12].weight)
        
    def forward(self, x):
        # x: [B, 128, 28, 28]
        x = self.net(x)  # [B, out_ch, 224, 224]
        return x
