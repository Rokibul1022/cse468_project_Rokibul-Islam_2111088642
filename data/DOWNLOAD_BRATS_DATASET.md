# BraTS 2023 Dataset Placeholder

## Dataset Information
- **Name:** ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
- **Size:** ~50GB (1,251 patient cases)
- **Format:** NIfTI (.nii.gz files)
- **Classes:** 4 (Background, Necrotic, Edema, Enhancing)

## Download Instructions

### 1. Register and Download
1. Visit: [BraTS Challenge](http://braintumorsegmentation.org/)
2. Register for BraTS 2023 Challenge
3. Download training data
4. Extract to this folder: `data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/`

### 2. Expected Structure
```
ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   └── BraTS-GLI-00000-000-seg.nii.gz
├── BraTS-GLI-00001-000/
└── ... (1,251 total cases)
```

### 3. Alternative: Use Sample Data
For testing purposes, you can use the sample images in `sample_mri_images/` folder.

## Usage in Project
- **Stage 1 (DINO):** Uses unlabeled images (no segmentation masks)
- **Stage 2 (STEGO):** Uses unlabeled images for clustering
- **Stage 3 (Fine-tuning):** Uses 10% of labeled data (with segmentation masks)

## File Size Warning
This dataset is ~50GB. Make sure you have sufficient disk space before downloading.