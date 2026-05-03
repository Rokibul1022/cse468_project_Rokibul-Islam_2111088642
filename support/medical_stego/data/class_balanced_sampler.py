"""
Class-Balanced Sampler
=======================
Oversamples rare tumor classes to handle severe class imbalance.

Problem:
- Background: 98% of pixels
- Necrotic: 0.19% (rarest)
- Edema: 0.87%
- Enhancing: 0.39%

Solution:
- Give higher sampling weight to images containing rare classes
- Necrotic samples: 5x weight
- Enhancing samples: 2x weight
- Edema samples: 1.5x weight
"""

import torch
from torch.utils.data import Sampler
import numpy as np


class ClassBalancedSampler(Sampler):
    """
    Samples data with higher probability for rare classes.
    
    Args:
        dataset: BraTSDataset instance
        oversample_factor: How much to oversample rare classes (default: 5)
    """
    
    def __init__(self, dataset, oversample_factor=5):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        
        # Analyze which samples contain rare classes
        print("Analyzing dataset for class-balanced sampling...")
        self.sample_weights = self._compute_sample_weights()
        
        print(f"Class-balanced sampler initialized:")
        print(f"  - Total samples: {len(self.sample_weights)}")
        print(f"  - Samples with necrotic: {np.sum(self.sample_weights >= oversample_factor)}")
        print(f"  - Samples with enhancing: {np.sum(self.sample_weights >= 2.0)}")
    
    def _compute_sample_weights(self):
        """
        Compute sampling weight for each image based on rare class presence.
        
        Returns:
            np.array: Weight for each sample
        """
        weights = np.ones(len(self.dataset))
        
        for idx in range(len(self.dataset)):
            # Load mask
            mask_path = str(self.dataset.image_files[idx]).replace('_image.npy', '_mask.npy')
            mask = np.load(mask_path)
            
            # Check which classes are present
            has_necrotic = np.any(mask == 1)
            has_edema = np.any(mask == 2)
            has_enhancing = np.any(mask == 3)
            
            # Assign weight based on rarest class present
            if has_necrotic:
                weights[idx] = self.oversample_factor  # 5x for necrotic (rarest)
            elif has_enhancing:
                weights[idx] = 2.0  # 2x for enhancing
            elif has_edema:
                weights[idx] = 1.5  # 1.5x for edema
            # else: weight stays 1.0 for background-only
        
        return weights
    
    def __iter__(self):
        """
        Generate indices for sampling with replacement based on weights.
        """
        # Sample with replacement using computed weights
        indices = torch.multinomial(
            torch.from_numpy(self.sample_weights).float(),
            num_samples=len(self.dataset),
            replacement=True
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset)
