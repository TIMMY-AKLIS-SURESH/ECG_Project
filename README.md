# 12-Lead ECG Classification & Pipeline

This repository contains the deep learning training and evaluation pipeline for 12-lead ECG signal classification using ConvNeXt-Atto architectures.

## 📁 Repository Structure

```text
ECG_Project/
├── train_combined.py                 # Main training script for combined ECG dataset
├── train_and_evaluate_20pct.py       # Training and 20% validation pipeline
├── evaluate_holdout.py               # Evaluation script for holdout test sets
├── best_convnext_atto_combined.pth   # Pre-trained model weights (Combined model)
├── best_convnext_atto_peak.pth       # Pre-trained model weights (Peak/best validation)
├── ECG_Cleaned_Dataset/              # Preprocessed & cleaned ECG signal records
├── ECG_Combined_Dataset/             # Unified combined dataset splits
├── Holdout_Test_Set/                 # Dedicated holdout evaluation set
└── Mendeley_20_Holdout/              # Mendeley 20% holdout test dataset
