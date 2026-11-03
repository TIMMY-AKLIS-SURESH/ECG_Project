# 12-Lead ECG Classification & Pipeline

An end-to-end deep learning framework using ConvNeXt-Atto architectures for multi-domain 12-lead ECG signal classification.

---

## 🔄 Pipeline Execution Flowchart

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       12-Lead ECG Processing Pipeline                       │
└─────────────────────────────────────────────────────────────────────────────┘

 [ STEP 1: Baseline Training ]
 ┌──────────────────────┐      ┌──────────────────────┐
 │ ECG_Cleaned_Dataset  │      │ Mendeley_20_Holdout  │
 └──────────┬───────────┘      └──────────┬───────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
         ┌───────────────────────────────────┐
         │   train_and_evaluate_20pct.py     │
         │   (80% Train / 20% Val Split)     │
         └─────────────────┬─────────────────┘
                           │
                           ▼
         ┌───────────────────────────────────┐
         │    best_convnext_atto_peak.pth    │
         └─────────────────┬─────────────────┘
                           │
                           │ (Foundation Baseline Weights)
                           ▼
 [ STEP 2: Multi-Domain Fine-Tuning ]
 ┌──────────────────────┐  │
 │ ECG_Combined_Dataset │  │
 │ (PTB-XL, Georgia,    │  │
 │  & Chapman signals)  │  │
 └──────────┬───────────┘  │
            │              │
            └──────┬───────┘
                   │
                   ▼
         ┌───────────────────────────────────┐
         │        train_combined.py          │
         │  (Multi-Domain Dataset Training)  │
         └─────────────────┬─────────────────┘
                           │
                           ▼
         ┌───────────────────────────────────┐
         │  best_convnext_atto_combined.pth  │
         └─────────────────┬─────────────────┘
                           │
                           │ (Final Combined Checkpoint)
                           ▼
 [ STEP 3: Holdout Evaluation ]
 ┌──────────────────────┐  │
 │  Holdout_Test_Set    │  │
 │  (Unseen Raw Set)    │  │
 └──────────┬───────────┘  │
            │              │
            └──────┬───────┘
                   │
                   ▼
         ┌───────────────────────────────────┐
         │        evaluate_holdout.py        │
         │     (7-View TTA Evaluation)       │
         └─────────────────┬─────────────────┘
                           │
                           ▼
         ┌───────────────────────────────────┐
         │  • Official Test Accuracy         │
         │  • Confusion Matrix Plot          │
         └───────────────────────────────────┘
