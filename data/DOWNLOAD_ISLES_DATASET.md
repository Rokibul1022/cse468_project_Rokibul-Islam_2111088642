# ISLES 2022 Dataset Placeholder

## Dataset Information
- **Name:** ISLES 2022 (Ischemic Stroke Lesion Segmentation)
- **Size:** ~15GB (250 patient cases)
- **Format:** NIfTI (.nii.gz files)
- **Classes:** 2 (Background, Lesion)

## Download Instructions

### 1. Register and Download
1. Visit: [ISLES Challenge](https://www.isles-challenge.org/)
2. Register for ISLES 2022 Challenge
3. Download training data
4. Extract to this folder: `data/ISLES-2022/`

### 2. Expected Structure
```
ISLES-2022/
├── rawdata/
│   ├── sub-strokecase0001/
│   │   └── ses-0001/
│   │       └── dwi/
│   │           ├── sub-strokecase0001_ses-0001_dwi.nii.gz
│   │           └── sub-strokecase0001_ses-0001_dwi.json
│   ├── sub-strokecase0002/
│   └── ... (250 total cases)
└── derivatives/
    └── labels/
        ├── sub-strokecase0001/
        │   └── ses-0001/
        │       └── sub-strokecase0001_ses-0001_msk.nii.gz
        └── ...
```

### 3. Alternative: Use Sample Data
For testing purposes, you can use the sample images in `sample_stroke_images/` folder.

## Usage in Project
- **Stage 4 (Transfer Learning):** Uses 10% of ISLES data for stroke lesion segmentation
- **3D Processing:** Converts 2D slices to 3D volumes (128×128×16 patches)
- **Cross-Disease Transfer:** Applies brain tumor knowledge to stroke detection

## File Size Warning
This dataset is ~15GB. Make sure you have sufficient disk space before downloading.

## Preprocessing Required
After download, run preprocessing scripts to convert to 2D slices:
```bash
python support/medical_stego/data/preprocess_isles.py
```