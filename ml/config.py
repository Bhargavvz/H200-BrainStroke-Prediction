"""
Brain Stroke Prediction System — Configuration
All hyperparameters, paths, and training settings.
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

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64                  # Adjust based on available VRAM
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

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
