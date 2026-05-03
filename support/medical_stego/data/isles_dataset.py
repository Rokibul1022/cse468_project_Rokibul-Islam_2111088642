"""
ISLES Dataset Loader
====================
Loads ISLES 2022 stroke lesion segmentation data.

Dataset Structure:
- Images: MRI slices (DWI, ADC, FLAIR modalities)
- Masks: Binary segmentation labels:
  * 0: Background (healthy brain tissue)
  * 1: Stroke lesion (ischemic region)

Difference from BraTS:
- BraTS: 4 classes (background + 3 tumor types)
- ISLES: 2 classes (background + lesion)
- BraTS: Tumor segmentation
- ISLES: Stroke lesion segmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class ISLESDataset(Dataset):
    """
    PyTorch Dataset for ISLES stroke lesion segmentation.
    
    Args:
        data_dir: Path to preprocessed data (e.g., 'data/isles_slices/train')
        transform: Optional data augmentation transforms
        return_mask: If True, returns (image, mask). If False, returns only image
    """
    
    def __init__(self, data_dir, transform=None, return_mask=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_mask = return_mask
        
        # Load all image files
        self.image_files = sorted(list(self.data_dir.glob('*_image.npy')))
        
        print(f"Loaded {len(self.image_files)} ISLES samples from {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image (grayscale, 224x224, float32, range [0,1])
        image_path = self.image_files[idx]
        image = np.load(image_path)  # Shape: (224, 224)
        
        # Convert to 3-channel (DINO expects RGB input)
        image = np.stack([image, image, image], axis=0)  # Shape: (3, 224, 224)
        image = torch.from_numpy(image).float()
        
        # Load mask if needed
        if self.return_mask:
            mask_path = str(image_path).replace('_image.npy', '_mask.npy')
            mask = np.load(mask_path)  # Shape: (224, 224), values: 0 or 1
            mask = torch.from_numpy(mask).long()
            
            # Apply transforms if provided
            if self.transform:
                image, mask = self.transform(image, mask)
            
            return image, mask
        else:
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return image
    
    def get_class_distribution(self):
        """
        Compute class distribution across entire dataset.
        
        Returns:
            dict: {class_id: pixel_count}
        """
        class_counts = {0: 0, 1: 0}
        
        print("Computing class distribution...")
        for idx in range(len(self)):
            mask_path = str(self.image_files[idx]).replace('_image.npy', '_mask.npy')
            mask = np.load(mask_path)
            
            class_counts[0] += np.sum(mask == 0)
            class_counts[1] += np.sum(mask == 1)
        
        return class_counts
