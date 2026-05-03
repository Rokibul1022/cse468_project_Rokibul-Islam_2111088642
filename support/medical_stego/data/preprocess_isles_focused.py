"""
Improved ISLES Preprocessing - Lesion-Focused
==============================================
Extract patches centered on lesions to help model learn tiny strokes.

Strategy:
1. Find slices with lesions
2. Extract 224x224 patches centered on lesion
3. Augment with rotations/flips
4. Balance dataset (50% lesion patches, 50% background)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
import argparse
from scipy import ndimage


def find_lesion_center(mask):
    """Find center of mass of lesion."""
    if mask.max() == 0:
        return None
    
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    
    center = coords.mean(axis=0).astype(int)
    return center


def extract_lesion_patch(image, mask, center, size=224):
    """Extract patch centered on lesion."""
    h, w = image.shape
    cy, cx = center
    
    # Calculate patch boundaries
    y1 = max(0, cy - size//2)
    y2 = min(h, cy + size//2)
    x1 = max(0, cx - size//2)
    x2 = min(w, cx + size//2)
    
    # Extract patch
    img_patch = image[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]
    
    # Pad if needed
    if img_patch.shape[0] < size or img_patch.shape[1] < size:
        img_patch = np.pad(img_patch, 
                          ((0, size-img_patch.shape[0]), (0, size-img_patch.shape[1])),
                          mode='constant')
        mask_patch = np.pad(mask_patch,
                           ((0, size-mask_patch.shape[0]), (0, size-mask_patch.shape[1])),
                           mode='constant')
    
    return img_patch, mask_patch


def augment_patch(image, mask):
    """Apply random augmentation."""
    # Random rotation
    angle = np.random.choice([0, 90, 180, 270])
    if angle > 0:
        image = ndimage.rotate(image, angle, reshape=False, order=1)
        mask = ndimage.rotate(mask, angle, reshape=False, order=0)
    
    # Random flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    return image, mask


def preprocess_isles_lesion_focused(
    data_dir='data/ISLES-2022/isles-2022',
    output_dir='data/isles_slices_focused',
    train_split=0.8,
    augment_factor=5
):
    """
    Preprocess ISLES with lesion-focused patches.
    
    Args:
        data_dir: Path to raw ISLES data
        output_dir: Where to save preprocessed data
        train_split: Fraction for training
        augment_factor: How many augmented versions per lesion patch
    """
    
    data_dir = Path(data_dir)
    derivatives_dir = data_dir / 'derivatives'
    output_dir = Path(output_dir)
    
    print("="*60)
    print("ISLES LESION-FOCUSED PREPROCESSING")
    print("="*60)
    print(f"Strategy: Extract patches centered on lesions")
    print(f"Augmentation: {augment_factor}x per lesion")
    print()
    
    # Find all cases
    case_dirs = sorted(list(data_dir.glob('sub-strokecase*/ses-*')))
    case_dirs = [d for d in case_dirs if 'derivatives' not in str(d)]
    
    # Split train/val
    num_train = int(len(case_dirs) * train_split)
    train_cases = case_dirs[:num_train]
    val_cases = case_dirs[num_train:]
    
    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    print()
    
    # Process training cases
    print("Processing training cases (lesion-focused)...")
    train_output = output_dir / 'train'
    train_output.mkdir(parents=True, exist_ok=True)
    
    train_lesion_patches = 0
    train_background_patches = 0
    
    for case_dir in train_cases:
        subject_name = case_dir.parent.name
        session_name = case_dir.name
        case_id = subject_name.replace('sub-strokecase', 'case')
        
        # Load FLAIR image
        image_path = case_dir / 'anat' / f"{subject_name}_{session_name}_FLAIR.nii"
        mask_path = derivatives_dir / subject_name / session_name / f"{subject_name}_{session_name}_msk.nii"
        
        if not image_path.exists() or not mask_path.exists():
            continue
        
        print(f"Processing {case_id}...")
        
        # Load volumes
        image_vol = nib.load(image_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()
        
        num_slices = min(image_vol.shape[2], mask_vol.shape[2])
        
        for slice_idx in range(num_slices):
            image_slice = image_vol[:, :, slice_idx]
            mask_slice = mask_vol[:, :, slice_idx]
            
            # Skip empty slices
            if image_slice.max() < 0.01:
                continue
            
            # Normalize image
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
            mask_slice = (mask_slice > 0).astype(np.uint8)
            
            # Check if slice has lesion
            has_lesion = mask_slice.max() > 0
            
            if has_lesion:
                # Extract lesion-centered patch
                center = find_lesion_center(mask_slice)
                if center is not None:
                    img_patch, mask_patch = extract_lesion_patch(image_slice, mask_slice, center)
                    
                    # Resize to 224x224
                    img_patch = np.array(Image.fromarray((img_patch * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                    mask_patch = np.array(Image.fromarray(mask_patch.astype(np.uint8)).resize((224, 224), Image.NEAREST))
                    
                    # Save original
                    slice_name = f"{case_id}_slice{slice_idx:03d}_lesion"
                    np.save(train_output / f"{slice_name}_image.npy", img_patch.astype(np.float32))
                    np.save(train_output / f"{slice_name}_mask.npy", mask_patch.astype(np.int64))
                    train_lesion_patches += 1
                    
                    # Save augmented versions
                    for aug_idx in range(augment_factor):
                        img_aug, mask_aug = augment_patch(img_patch.copy(), mask_patch.copy())
                        aug_name = f"{case_id}_slice{slice_idx:03d}_lesion_aug{aug_idx}"
                        np.save(train_output / f"{aug_name}_image.npy", img_aug.astype(np.float32))
                        np.save(train_output / f"{aug_name}_mask.npy", mask_aug.astype(np.int64))
                        train_lesion_patches += 1
            
            else:
                # Save some background patches for balance
                if np.random.rand() < 0.1:  # 10% of background slices
                    img_resized = np.array(Image.fromarray((image_slice * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                    mask_resized = np.zeros((224, 224), dtype=np.int64)
                    
                    slice_name = f"{case_id}_slice{slice_idx:03d}_bg"
                    np.save(train_output / f"{slice_name}_image.npy", img_resized.astype(np.float32))
                    np.save(train_output / f"{slice_name}_mask.npy", mask_resized)
                    train_background_patches += 1
    
    print(f"\nTrain patches:")
    print(f"  Lesion: {train_lesion_patches}")
    print(f"  Background: {train_background_patches}")
    print(f"  Total: {train_lesion_patches + train_background_patches}")
    print()
    
    # Process validation cases (no augmentation)
    print("Processing validation cases...")
    val_output = output_dir / 'val'
    val_output.mkdir(parents=True, exist_ok=True)
    
    val_lesion_patches = 0
    val_background_patches = 0
    
    for case_dir in val_cases:
        subject_name = case_dir.parent.name
        session_name = case_dir.name
        case_id = subject_name.replace('sub-strokecase', 'case')
        
        image_path = case_dir / 'anat' / f"{subject_name}_{session_name}_FLAIR.nii"
        mask_path = derivatives_dir / subject_name / session_name / f"{subject_name}_{session_name}_msk.nii"
        
        if not image_path.exists() or not mask_path.exists():
            continue
        
        print(f"Processing {case_id}...")
        
        image_vol = nib.load(image_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()
        
        num_slices = min(image_vol.shape[2], mask_vol.shape[2])
        
        for slice_idx in range(num_slices):
            image_slice = image_vol[:, :, slice_idx]
            mask_slice = mask_vol[:, :, slice_idx]
            
            if image_slice.max() < 0.01:
                continue
            
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
            mask_slice = (mask_slice > 0).astype(np.uint8)
            
            has_lesion = mask_slice.max() > 0
            
            if has_lesion:
                center = find_lesion_center(mask_slice)
                if center is not None:
                    img_patch, mask_patch = extract_lesion_patch(image_slice, mask_slice, center)
                    img_patch = np.array(Image.fromarray((img_patch * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                    mask_patch = np.array(Image.fromarray(mask_patch.astype(np.uint8)).resize((224, 224), Image.NEAREST))
                    
                    slice_name = f"{case_id}_slice{slice_idx:03d}_lesion"
                    np.save(val_output / f"{slice_name}_image.npy", img_patch.astype(np.float32))
                    np.save(val_output / f"{slice_name}_mask.npy", mask_patch.astype(np.int64))
                    val_lesion_patches += 1
            else:
                if np.random.rand() < 0.1:
                    img_resized = np.array(Image.fromarray((image_slice * 255).astype(np.uint8)).resize((224, 224))) / 255.0
                    mask_resized = np.zeros((224, 224), dtype=np.int64)
                    
                    slice_name = f"{case_id}_slice{slice_idx:03d}_bg"
                    np.save(val_output / f"{slice_name}_image.npy", img_resized.astype(np.float32))
                    np.save(val_output / f"{slice_name}_mask.npy", mask_resized)
                    val_background_patches += 1
    
    print(f"\nVal patches:")
    print(f"  Lesion: {val_lesion_patches}")
    print(f"  Background: {val_background_patches}")
    print(f"  Total: {val_lesion_patches + val_background_patches}")
    print()
    
    print("="*60)
    print("LESION-FOCUSED PREPROCESSING COMPLETED")
    print(f"Output: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ISLES-2022/isles-2022')
    parser.add_argument('--output_dir', default='data/isles_slices_focused')
    parser.add_argument('--augment_factor', type=int, default=5)
    
    args = parser.parse_args()
    
    preprocess_isles_lesion_focused(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor
    )
