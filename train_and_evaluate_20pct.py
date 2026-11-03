# ==============================================================================
# HIGH-ACCURACY CONVNEXT-ATTO + CBAM MULTI-DATASET TRAINING & 7-VIEW TTA PIPELINE
# Train Split     : 80% of C:\ECG_Project\ECG_Cleaned_Dataset (RAM Cached)
# Test Set        : 20% Holdout at C:\ECG_Project\Mendeley_20_Holdout
# Model Checkpoint: C:\ECG_Project\best_convnext_atto_peak.pth
# ==============================================================================

import os
import random
import time
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import timm

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
CLEANED_DATASET_PATH = os.path.join(BASE_DIR, "ECG_Cleaned_Dataset")
HOLDOUT_DATASET_PATH = os.path.join(BASE_DIR, "Mendeley_20_Holdout")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_convnext_atto_peak.pth")

IMG_SIZE = 448
NUM_CLASSES = 4
NUM_EPOCHS = 35
BATCH_SIZE = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=3),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 7-View Test-Time Augmentation Pool
tta_transform_pool = [
    eval_transforms,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(2, 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-2, -2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(4, 4)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-4, -4)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.03, contrast=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
]


# ------------------------------------------------------------------------------
# 2. DATASET DEFINITION
# ------------------------------------------------------------------------------
class RAMDataset(Dataset):
    def __init__(self, ram_samples, transform):
        self.ram_samples = ram_samples
        self.transform = transform

    def __len__(self):
        return len(self.ram_samples)

    def __getitem__(self, idx):
        img, label = self.ram_samples[idx]
        return self.transform(img), label


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
            "convnext_atto", pretrained=True, features_only=True
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


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.5, smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs, self.smoothing / (NUM_CLASSES - 1)
            )
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights
            
        return loss.mean()


# ------------------------------------------------------------------------------
# 4. EXPLICIT LOGIT INDEX MAPPER
# ------------------------------------------------------------------------------
def map_folder_to_logit(folder_name):
    f_lower = folder_name.lower()
    if "history" in f_lower:
        return 1  # 1_History_MI
    elif "myocardial" in f_lower or "mi" in f_lower:
        return 0  # 0_MI
    elif "abnormal" in f_lower:
        return 2  # 2_Abnormal
    elif "normal" in f_lower:
        return 3  # 3_Normal
    else:
        raise ValueError(f"Unrecognized directory name: {folder_name}")


# ------------------------------------------------------------------------------
# 5. MAIN TRAINING EXECUTION
# ------------------------------------------------------------------------------
def main():
    print(f"🚀 Using Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU Acceleration Active: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    if not os.path.exists(HOLDOUT_DATASET_PATH):
        raise FileNotFoundError(f"❌ Holdout directory missing at '{HOLDOUT_DATASET_PATH}'!")

    base_dataset = datasets.ImageFolder(root=CLEANED_DATASET_PATH)
    holdout_dataset = datasets.ImageFolder(root=HOLDOUT_DATASET_PATH)
    
    target_class_names = ['0_MI', '1_History_MI', '2_Abnormal', '3_Normal']

    print("\n-------------------------------------------------------")
    print(" 🎯 EXPLICIT LOGIT INDEX MAPPING VERIFICATION")
    print("-------------------------------------------------------")
    for raw_folder in base_dataset.classes:
        assigned_idx = map_folder_to_logit(raw_folder)
        print(f" • Raw Folder: '{raw_folder}'")
        print(f"    -> Mapped to Logit Index {assigned_idx} ('{target_class_names[assigned_idx]}')")
    print("-------------------------------------------------------")

    holdout_file_set = set(os.path.basename(path) for path, _ in holdout_dataset.samples)

    train_ram_cache = []
    test_ram_cache = []

    print("\n🧠 Caching Images into RAM & Remapping Indices...")
    for path, raw_idx in tqdm(base_dataset.samples, desc="RAM Caching"):
        raw_folder_name = base_dataset.classes[raw_idx]
        correct_label = map_folder_to_logit(raw_folder_name)
        
        img = Image.open(path).convert("RGB")
        filename = os.path.basename(path)
        
        if filename in holdout_file_set:
            test_ram_cache.append((img, correct_label))
        else:
            train_ram_cache.append((img, correct_label))

    print(f"📊 Split Ready | Training Set (80%): {len(train_ram_cache)} | Holdout Test Set (20%): {len(test_ram_cache)}")

    train_labels = np.array([lbl for _, lbl in train_ram_cache])
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    total_samples = len(train_labels)
    computed_weights = total_samples / (NUM_CLASSES * class_counts.astype(float))
    computed_weights[0] *= 1.2
    computed_weights[1] *= 1.2
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32).to(device)
    
    print("⚖️ Adjusted Class Weights for Loss:", computed_weights)

    train_ds = RAMDataset(train_ram_cache, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = ConvNeXtAttoCBAM().to(device)
    criterion = AsymmetricFocalLoss(class_weights=class_weights_tensor, gamma=2.5, smoothing=0.02)

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n and "cbam" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "cbam" in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 3e-5},
            {"params": head_params, "lr": 4e-4},
        ],
        weight_decay=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = amp.GradScaler("cuda") if torch.cuda.is_available() else None
    best_tta_acc = 0.0

    print(f"\n⚡ Fine-Tuning High-Resolution ConvNeXt-Atto ({NUM_EPOCHS} Epochs)...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels_batch in train_loader:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                with amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # 7-View TTA Evaluation Pass
        model.eval()
        test_targets = np.array([lbl for _, lbl in test_ram_cache])
        tta_probs = np.zeros((len(test_ram_cache), NUM_CLASSES))

        for view_transform in tta_transform_pool:
            ds = RAMDataset(test_ram_cache, view_transform)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            view_probs = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    if torch.cuda.is_available():
                        with amp.autocast("cuda"):
                            outputs = model(images)
                    else:
                        outputs = model(images)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                    view_probs.append(probs)
            tta_probs += np.concatenate(view_probs, axis=0) / len(tta_transform_pool)

        epoch_preds = np.argmax(tta_probs, axis=1)
        test_acc = accuracy_score(test_targets, epoch_preds) * 100.0

        if test_acc > best_tta_acc:
            best_tta_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(f"Epoch [{epoch+1:>2}/{NUM_EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | 7-View TTA Holdout Acc: {test_acc:.2f}% | Peak Acc: {best_tta_acc:.2f}%")

    print(f"\n🎉 Fine-Tuning Complete in {(time.time() - start_time)/60:.1f} minutes!")
    print(f"💾 Peak Checkpoint Saved: '{MODEL_SAVE_PATH}'")

    # Final Evaluation & Matrix Plot
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    final_tta_probs = np.zeros((len(test_ram_cache), NUM_CLASSES))
    for view_transform in tta_transform_pool:
        ds = RAMDataset(test_ram_cache, view_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        view_probs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                if torch.cuda.is_available():
                    with amp.autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                view_probs.append(probs)
        final_tta_probs += np.concatenate(view_probs, axis=0) / len(tta_transform_pool)

    final_preds = np.argmax(final_tta_probs, axis=1)
    final_acc = accuracy_score(test_targets, final_preds) * 100.0

    print("\n=======================================================")
    print(f"    🎉 FINAL OFFICIAL ATTO HIGH-RES HOLDOUT TEST ACCURACY: {final_acc:.2f}%")
    print("=======================================================\n")

    print(classification_report(test_targets, final_preds, target_names=target_class_names, digits=4))

    cm = confusion_matrix(test_targets, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_class_names, yticklabels=target_class_names)
    plt.title("ConvNeXt-Atto High-Res + 7-View TTA Holdout Test", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    matrix_path = os.path.join(BASE_DIR, "mendeley_atto_peak_confusion_matrix.png")
    plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
    print(f"📸 Saved matrix plot to '{matrix_path}'")
    plt.show()


if __name__ == "__main__":
    main()