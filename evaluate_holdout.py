# ==============================================================================
# HIGH-ACCURACY MULTI-DATASET 7-VIEW TTA HOLDOUT EVALUATION PIPELINE
# Evaluation Path : C:\ECG_Project\Holdout_Test_Set (2,000 images)
# Model Checkpoint: C:\ECG_Project\best_convnext_atto_combined.pth
# ==============================================================================

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import timm
from PIL import Image

# ------------------------------------------------------------------------------
# 1. SETUP & REPRODUCIBILITY
# ------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = r"C:\ECG_Project"
TEST_DIR = os.path.join(BASE_DIR, "Holdout_Test_Set")
# ✅ Fixed to point to the multi-dataset combined checkpoint
MODEL_PATH = os.path.join(BASE_DIR, "best_convnext_atto_combined.pth")

IMG_SIZE = 448
NUM_CLASSES = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------------------------
# 2. 7-VIEW TTA TRANSFORM POOL (High-Res Compatible)
# ------------------------------------------------------------------------------
base_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(2, 2)),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-2, -2)),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(4, 4)),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-4, -4)),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        base_norm,
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.03, contrast=0.03),
        transforms.ToTensor(),
        base_norm,
    ]),
]

# ------------------------------------------------------------------------------
# 3. ARCHITECTURE DEFINITION (ConvNeXt-Atto + CBAM)
# ------------------------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, planes, ratio=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(planes // ratio, planes, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_out = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * sa_out


class ConvNeXtAttoCBAM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_atto", pretrained=False, features_only=True
        )
        in_features = self.backbone.feature_info[-1]["num_chs"]
        self.cbam = CBAM(planes=in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        x = self.cbam(feats)
        x = self.pool(x)
        return self.classifier(x)

# ------------------------------------------------------------------------------
# 4. DATASET DEFINITIONS
# ------------------------------------------------------------------------------
class RawDiskDataset(Dataset):
    def __init__(self, dataset_folder):
        self.base_ds = datasets.ImageFolder(root=dataset_folder)
        self.samples = self.base_ds.samples
        self.classes = self.base_ds.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, label


class ViewDataset(Dataset):
    def __init__(self, raw_ds, transform):
        self.raw_ds = raw_ds
        self.transform = transform

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, i):
        img, lbl = self.raw_ds[i]
        return self.transform(img), lbl

# ------------------------------------------------------------------------------
# 5. MAIN EVALUATION EXECUTION
# ------------------------------------------------------------------------------
def main():
    print(f"🚀 Hardware Device: {device}", flush=True)
    
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"❌ Holdout directory missing at '{TEST_DIR}'")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model checkpoint missing at '{MODEL_PATH}'")

    dataset = RawDiskDataset(TEST_DIR)
    class_names = dataset.classes
    num_samples = len(dataset)

    print("\n-------------------------------------------------------", flush=True)
    print(" 🎯 HOLDOUT DATASET LOGIT MAPPING & VERIFICATION", flush=True)
    print("-------------------------------------------------------", flush=True)
    for idx, cls_name in enumerate(class_names):
        print(f" • Logit Index {idx} -> '{cls_name}'", flush=True)
    print("-------------------------------------------------------", flush=True)

    assert class_names == ['0_MI', '1_History_MI', '2_Abnormal', '3_Normal'], \
        "ERROR: Indexing mismatch!"

    model = ConvNeXtAttoCBAM().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n⚡ Running Multi-Dataset High-Res 7-View TTA Evaluation on {num_samples} holdout samples...", flush=True)

    accumulated_probs = np.zeros((num_samples, NUM_CLASSES))
    true_targets = []

    for v_idx, t_func in enumerate(tta_transforms, 1):
        print(f" 🔄 Processing TTA View {v_idx}/7...", flush=True)
        view_probs = []
        targets_list = []

        v_loader = DataLoader(ViewDataset(dataset, t_func), batch_size=8, shuffle=False, num_workers=0)

        with torch.no_grad():
            for images, labels in v_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                view_probs.append(probs)
                if v_idx == 1:
                    targets_list.extend(labels.numpy())

        accumulated_probs += np.vstack(view_probs)
        if v_idx == 1:
            true_targets = np.array(targets_list)

    final_probs = accumulated_probs / len(tta_transforms)
    final_preds = np.argmax(final_probs, axis=1)

    acc = accuracy_score(true_targets, final_preds) * 100.0

    print("\n=======================================================", flush=True)
    print(f"    🎉 MULTI-DATASET CONVNEXT-ATTO 7-VIEW TTA ACCURACY: {acc:.2f}%", flush=True)
    print("=======================================================\n", flush=True)
    print(classification_report(true_targets, final_preds, target_names=class_names, digits=4))
    
    print("\n🧩 Confusion Matrix:", flush=True)
    cm = confusion_matrix(true_targets, final_preds)
    print(f"{'':<15} Pred 0_MI  Pred 1_Hist_MI Pred 2_Abnormal Pred 3_Normal")
    for idx, row in enumerate(cm):
        print(f"True {class_names[idx]:<12} {row[0]:<12} {row[1]:<16} {row[2]:<16} {row[3]:<12}")

if __name__ == "__main__":
    main()