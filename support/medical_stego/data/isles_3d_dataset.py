"""
3D ISLES Dataset Loader
========================
Loads full 3D MRI volumes for stroke lesion segmentation.

Key differences from 2D:
- Loads entire volume (H, W, D) not single slices
- Preserves 3D context for tiny lesions
- Uses patch-based training (extract 3D patches)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
import random


class ISLES3DDataset(Dataset):
    """
    3D Dataset for ISLES stroke lesion segmentation.
    
    Args:
        data_dir: Path to raw ISLES data
        patch_size: Size of 3D patches (default: 128x128x16)
        patches_per_volume: How many patches to extract per volume
        return_mask: If True, returns (volume, mask)
    """
    
    def __init__(self, data_dir, derivatives_dir, patch_size=(128, 128, 16), 
                 patches_per_volume=4, return_mask=True, case_dirs=None):
        self.data_dir = Path(data_dir)
        self.derivatives_dir = Path(derivatives_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.return_mask = return_mask
        self.volume_cache = {}  # Cache loaded volumes
        
        # If case_dirs provided, use them directly (already validated)
        if case_dirs is not None:
            self.case_dirs = case_dirs
            print(f"Using {len(self.case_dirs)} provided case directories")
        else:
            # Find all cases
            all_case_dirs = sorted(list(self.data_dir.glob('sub-strokecase*/ses-*')))
            all_case_dirs = [d for d in all_case_dirs if 'derivatives' not in str(d)]
            
            # Filter valid volumes (no zero dimensions)
            self.case_dirs = []
            for case_dir in all_case_dirs:
                vol, mask = self.load_volume(case_dir)
                if vol is not None and vol.size > 0:
                    if 0 not in vol.shape and 0 not in mask.shape:
                        self.case_dirs.append(case_dir)
                        # Cache the volume
                        self.volume_cache[str(case_dir)] = (vol, mask)
            
            print(f"Loaded {len(self.case_dirs)} valid 3D volumes from {data_dir} (filtered from {len(all_case_dirs)})")
    
    def __len__(self):
        return len(self.case_dirs) * self.patches_per_volume
    
    def load_volume(self, case_dir):
        """Load full 3D volume and mask."""
        subject_name = case_dir.parent.name
        session_name = case_dir.name
        
        # Load FLAIR image
        image_path = case_dir / 'anat' / f"{subject_name}_{session_name}_FLAIR.nii"
        mask_path = self.derivatives_dir / subject_name / session_name / f"{subject_name}_{session_name}_msk.nii"
        
        if not image_path.exists() or not mask_path.exists():
            return None, None
        
        try:
            # Load volumes
            image_vol = nib.load(image_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()
            
            # Check if volumes are valid
            if image_vol.size == 0 or mask_vol.size == 0:
                return None, None
            
            # Check for zero dimensions
            if 0 in image_vol.shape or 0 in mask_vol.shape:
                return None, None
            
            # Ensure same spatial size
            min_h = min(image_vol.shape[0], mask_vol.shape[0])
            min_w = min(image_vol.shape[1], mask_vol.shape[1])
            min_d = min(image_vol.shape[2], mask_vol.shape[2])
            
            image_vol = image_vol[:min_h, :min_w, :min_d]
            mask_vol = mask_vol[:min_h, :min_w, :min_d]
            
            # Safety check
            assert image_vol.shape == mask_vol.shape, \
                f"Shape mismatch: image {image_vol.shape}, mask {mask_vol.shape}"
            
            # Check again after slicing
            if 0 in image_vol.shape or 0 in mask_vol.shape:
                return None, None
            
            # Normalize image
            image_vol = (image_vol - image_vol.min()) / (image_vol.max() - image_vol.min() + 1e-8)
            mask_vol = (mask_vol > 0).astype(np.int64)
            
            return image_vol, mask_vol
        except Exception as e:
            print(f"Error loading {subject_name}: {e}")
            return None, None
    
    def extract_patch(self, volume, mask, patch_size, center_on_lesion=True):
        """
        Robust 3D patch extraction with automatic padding.
        Always returns exact patch_size.
        """
        ph, pw, pd = patch_size
        H, W, D = volume.shape
        
        # Pad volume if smaller than patch
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad_d = max(0, pd - D)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            volume = np.pad(volume, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='edge')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            H, W, D = volume.shape
        
        # Choose center point
        if center_on_lesion and mask.max() > 0:
            lesion_coords = np.argwhere(mask > 0)
            if len(lesion_coords) > 0:
                cy, cx, cz = map(int, lesion_coords[random.randint(0, len(lesion_coords) - 1)])
            else:
                cy, cx, cz = H // 2, W // 2, D // 2
        else:
            # Random location for diverse background patches
            cy = random.randint(ph // 2, max(ph // 2, H - ph // 2))
            cx = random.randint(pw // 2, max(pw // 2, W - pw // 2))
            cz = random.randint(pd // 2, max(pd // 2, D - pd // 2))
        
        # Compute valid start coordinates
        y1 = max(0, min(cy - ph // 2, H - ph))
        x1 = max(0, min(cx - pw // 2, W - pw))
        z1 = max(0, min(cz - pd // 2, D - pd))
        
        # End coordinates
        y2 = y1 + ph
        x2 = x1 + pw
        z2 = z1 + pd
        
        # Extract exact-size patch
        patch_vol = volume[y1:y2, x1:x2, z1:z2]
        patch_mask = mask[y1:y2, x1:x2, z1:z2]
        
        # Final safety check
        assert patch_vol.shape == patch_size, f"Bad volume patch shape: {patch_vol.shape}"
        assert patch_mask.shape == patch_size, f"Bad mask patch shape: {patch_mask.shape}"
        
        return patch_vol.astype(np.float32), patch_mask.astype(np.int64)
    
    def __getitem__(self, idx):
        # Determine which volume and which patch
        volume_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        case_dir = self.case_dirs[volume_idx]
        
        # Load volume from cache or disk
        cache_key = str(case_dir)
        if cache_key in self.volume_cache:
            volume, mask = self.volume_cache[cache_key]
        else:
            volume, mask = self.load_volume(case_dir)
            # Validate and cache
            if volume is not None and 0 not in volume.shape and 0 not in mask.shape:
                self.volume_cache[cache_key] = (volume, mask)
            else:
                # Return zero patch for invalid volume
                patch_vol = np.zeros(self.patch_size, dtype=np.float32)
                patch_mask = np.zeros(self.patch_size, dtype=np.int64)
                patch_vol = torch.from_numpy(patch_vol).unsqueeze(0)
                patch_mask = torch.from_numpy(patch_mask)
                return patch_vol, patch_mask if self.return_mask else patch_vol
        
        # Extract patch (center on lesion for first half of patches)
        center_on_lesion = (patch_idx < self.patches_per_volume // 2)
        patch_vol, patch_mask = self.extract_patch(volume, mask, self.patch_size, center_on_lesion)
        
        # FORCE exact size - this is critical
        assert patch_vol.shape == self.patch_size, f"Volume shape {patch_vol.shape} != {self.patch_size}"
        assert patch_mask.shape == self.patch_size, f"Mask shape {patch_mask.shape} != {self.patch_size}"
        
        # Ensure contiguous arrays with proper memory layout
        patch_vol = np.ascontiguousarray(patch_vol, dtype=np.float32)
        patch_mask = np.ascontiguousarray(patch_mask, dtype=np.int64)
        
        # Convert to tensor
        patch_vol = torch.from_numpy(patch_vol).unsqueeze(0)  # (1, H, W, D)
        patch_mask = torch.from_numpy(patch_mask)  # (H, W, D)
        
        if self.return_mask:
            return patch_vol, patch_mask
        else:
            return patch_vol
