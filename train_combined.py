# ==============================================================================
# HIGH-PERFORMANCE END-TO-END MULTI-DOMAIN FINE-TUNING PIPELINE
# Warm-Start Path: C:\ECG_Project\best_convnext_atto_peak.pth
# Save Path      : C:\ECG_Project\best_convnext_atto_combined.pth
# ==============================================================================

import os
import random
import time
import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm
from PIL import Image
from tqdm import tqdm

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
COMBINED_DIR = os.path.join(BASE_DIR, "ECG_Combined_Dataset")
WARMSTART_PEAK_PATH = os.path.join(BASE_DIR, "best_convnext_atto_peak.pth")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_convnext_atto_combined.pth")

IMG_SIZE = 448
NUM_CLASSES = 4
BATCH_SIZE = 8
ACCUM_STEPS = 2
EPOCHS = 25

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=(-3, 3)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ------------------------------------------------------------------------------
# 2. ARCHITECTURE DEFINITION (ConvNeXt-Atto + CBAM)
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
# 3. DATASET DEFINITION & SPLIT
# ------------------------------------------------------------------------------
class CustomTransformDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def main():
    print(f"🚀 Hardware Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"🎮 GPU Active: {torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.empty_cache()
    
    if not os.path.exists(COMBINED_DIR):
        raise FileNotFoundError(f"❌ Directory missing at '{COMBINED_DIR}'")

    full_ds = datasets.ImageFolder(root=COMBINED_DIR)
    samples = full_ds.samples
    class_names = full_ds.classes
    labels = np.array([s[1] for s in samples])

    print("\n-------------------------------------------------------", flush=True)
    print(" 🎯 TRAINING CORPUS CLASS DISTRIBUTION & LOGIT MAPPING", flush=True)
    print("-------------------------------------------------------", flush=True)
    for c_idx, c_name in enumerate(class_names):
        count = (labels == c_idx).sum()
        print(f" • Logit Index {c_idx} -> '{c_name:<15}': {count} images", flush=True)
    print("-------------------------------------------------------", flush=True)

    assert class_names == ['0_MI', '1_History_MI', '2_Abnormal', '3_Normal'], \
        "ERROR: Indexing mismatch!"

    rng = np.random.RandomState(SEED)
    all_idx = np.arange(len(samples))
    val_mask = np.zeros(len(samples), dtype=bool)

    for c in range(NUM_CLASSES):
        c_indices = all_idx[labels == c]
        rng.shuffle(c_indices)
        n_val = int(0.15 * len(c_indices))
        val_mask[c_indices[:n_val]] = True

    train_samples = [samples[i] for i in np.where(~val_mask)[0]]
    val_samples = [samples[i] for i in np.where(val_mask)[0]]

    print(f"📊 Split Ready | Training: {len(train_samples)} | Validation: {len(val_samples)}", flush=True)

    train_loader = DataLoader(CustomTransformDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(CustomTransformDataset(val_samples, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNeXtAttoCBAM().to(device)
    
    if os.path.exists(WARMSTART_PEAK_PATH):
        peak_weights = torch.load(WARMSTART_PEAK_PATH, map_location=device, weights_only=True)
        model.load_state_dict(peak_weights, strict=True)
        print(f"✅ Loaded 100% Peak baseline weights from '{WARMSTART_PEAK_PATH}' (Strict Match)", flush=True)

    # Stable CrossEntropy with light label smoothing to prevent overconfidence on noisy combined domains
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Uniform end-to-end low learning rates to gently adapt weights without breaking peak features
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.cbam.parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = amp.GradScaler("cuda") if torch.cuda.is_available() else None

    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n⚡ Starting End-to-End Fine-Tuning ({EPOCHS} Epochs)...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:2d}/{EPOCHS}]", leave=False)

        for step, (images, targets) in enumerate(pbar):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if torch.cuda.is_available():
                with amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets) / ACCUM_STEPS
                scaler.scale(loss).backward()
                
                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets) / ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.4f}"})

        scheduler.step()

        # Validation Pass
        model.eval()
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if torch.cuda.is_available():
                    with amp.autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_acc = (val_correct / total_val) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_str = " (Saved Peak Checkpoint)"
        else:
            save_str = ""

        print(f"Epoch [{epoch:2d}/{EPOCHS}] | Val Acc: {val_acc:.2f}% | Peak Val: {best_val_acc:.2f}%{save_str}", flush=True)

    total_time = time.time() - start_time
    print(f"\n🎉 Fine-Tuning Complete in {total_time:.1f}s!", flush=True)
    print(f"💾 Saved Checkpoint: '{MODEL_SAVE_PATH}'", flush=True)

if __name__ == "__main__":
    main()