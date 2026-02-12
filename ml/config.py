"""
Brain Stroke Prediction System — Configuration
All hyperparameters, paths, and training settings.

Training Strategy (2-Phase Fine-Tuning):
  Phase 1 — Freeze backbone, train only classification heads
  Phase 2 — Unfreeze backbone with discriminative LR (10x lower for backbone)
"""
import os
import torch

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

for d in [DATA_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ───────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Ischemia", "Bleeding"]
NUM_CLASSES = len(CLASS_NAMES)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_SIZE = 380                   # EfficientNet-B4 optimal input
IMG_CHANNELS = 3                 # Convert grayscale → 3-channel for pretrained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Training — Phase 1 (Frozen Backbone) ─────────────────────────────────────
BATCH_SIZE = 64                  # Large batch for H200 GPU
NUM_WORKERS = 4
PIN_MEMORY = True
PHASE1_EPOCHS = 10               # Train head only
PHASE1_LR = 1e-3                 # Higher LR when backbone is frozen

# ─── Training — Phase 2 (Full Fine-Tuning) ────────────────────────────────────
PHASE2_EPOCHS = 40               # Full fine-tuning
PHASE2_LR_BACKBONE = 1e-5        # Very low LR for pretrained backbone (10x lower)
PHASE2_LR_HEAD = 1e-4            # Regular LR for classification head
WARMUP_EPOCHS = 3                # Linear LR warmup at start of Phase 2

# ─── General Training ─────────────────────────────────────────────────────────
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS   # 50 total
EARLY_STOP_PATIENCE = 12
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0

# ─── Mixed Precision ──────────────────────────────────────────────────────────
USE_AMP = True
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ─── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Clinical Features ────────────────────────────────────────────────────────
CLINICAL_FEATURES = [
    "age", "gender", "hypertension", "heart_disease",
    "avg_glucose_level", "bmi", "smoking_status", "cholesterol"
]
NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)

# ─── Model Architecture ───────────────────────────────────────────────────────
IMAGE_FEATURE_DIM = 256
CLINICAL_FEATURE_DIM = 16
FUSION_HIDDEN_DIM = 128
DROPOUT_RATE = 0.4
DROPOUT_RATE_FC = 0.3

# ─── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_T0 = 10
SCHEDULER_TMULT = 2

# ─── Random Seed ───────────────────────────────────────────────────────────────
SEED = 42
