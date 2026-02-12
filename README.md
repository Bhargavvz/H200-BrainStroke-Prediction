# 🧠 Brain Stroke Prediction — Hybrid Deep Learning System

**End-to-end Brain Stroke Prediction** using Brain MRI scans with a hybrid model combining **EfficientNet-B4** (image CNN) + **Clinical DNN** (tabular data) with **late fusion**, served via FastAPI backend with React+Vite frontend.

---

## 📂 Project Structure

```
Sparsha-major-final!/
├── ml/                          # Machine Learning Pipeline
│   ├── config.py                # Hyperparameters & settings
│   ├── download_data.py         # Kaggle dataset download
│   ├── preprocess.py            # MRI image transforms
│   ├── dataset.py               # PyTorch Dataset & DataLoaders
│   ├── models.py                # EfficientNet-B4, ClinicalDNN, HybridFusion
│   ├── train.py                 # Training with mixed precision
│   ├── evaluate.py              # Metrics & visualizations
│   ├── grad_cam.py              # Grad-CAM explainability
│   └── requirements.txt
├── backend/                     # FastAPI Backend API
│   ├── main.py                  # API endpoints
│   ├── model_service.py         # Model loading & inference
│   ├── schemas.py               # Request/response schemas
│   └── requirements.txt
├── frontend/                    # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx              # Main app
│   │   ├── index.css            # Premium dark UI
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── MRIUpload.jsx
│   │   │   ├── ClinicalForm.jsx
│   │   │   ├── PredictionResult.jsx
│   │   │   ├── ConfidenceGauge.jsx
│   │   │   └── GradCAMView.jsx
│   │   └── api/predict.js
│   └── package.json
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (optional, CPU works too)
- Kaggle account (for dataset download)

### 1. ML Pipeline — Train the Model

```bash
# Install Python dependencies
cd ml
pip install -r requirements.txt

# Configure Kaggle API (if not already done)
# Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME + KAGGLE_KEY

# Download dataset
python download_data.py

# Train the hybrid model
python train.py

# Generate evaluation plots & Grad-CAM samples
python evaluate.py
```

### 2. Backend — Start the API Server

```bash
cd backend
pip install -r requirements.txt
python main.py
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 3. Frontend — Start the UI

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## 🧠 Model Architecture

### Hybrid Fusion Model (Late Fusion)

```
┌──────────────────┐    ┌──────────────────┐
│ Brain MRI Image   │    │ Clinical Features │
│ (380×380×3)       │    │ (age, BP, BMI...) │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
    EfficientNet-B4         4-Layer MLP
    (ImageNet pretrained)   (BatchNorm+ReLU)
         │                       │
    256-dim features        16-dim features
         │                       │
         └──────────┬────────────┘
                    │
              Concatenate (272-dim)
                    │
              Fusion Head (FC → 128 → 3)
                    │
            ┌───────┼───────┐
            │       │       │
         Normal  Ischemic  Hemorrhagic
```

### Why This Architecture?
- **EfficientNet-B4**: Best accuracy-to-parameter ratio; compound scaling captures multi-scale features critical for stroke lesion detection
- **Late Fusion**: Image and clinical data have fundamentally different feature spaces; late fusion allows each branch to learn domain-specific representations before combining
- **Transfer Learning**: ImageNet pretrained weights provide excellent initial features for medical imaging

---

## 📊 Dataset

- **Source**: Kaggle Brain Stroke CT Image Dataset
- **Classes**: Normal, Ischemic Stroke, Hemorrhagic Stroke
- **Size**: ~2,500 brain CT/MRI axial-slice images
- **Split**: 70% train / 15% validation / 15% test (stratified)

### Preprocessing
- Resize to 380×380
- Intensity normalization (ImageNet mean/std)
- Grayscale → 3-channel conversion
- Training augmentation: flip, rotate, color jitter, random erasing

---

## 📈 Training Features
- Mixed precision (BF16/FP16) for GPU acceleration
- Cosine annealing LR scheduler with warm restarts
- Class-weighted cross-entropy with label smoothing
- Early stopping (patience=10, monitored by AUC-ROC)
- Weighted random sampling for class imbalance

---

## 📊 Generated Visualizations
After running `evaluate.py`, these plots are saved to `ml/plots/`:
- Confusion Matrix
- ROC-AUC Curves (per-class)
- Precision-Recall Curves
- Training vs Validation Loss & Accuracy
- Per-class F1 Scores
- Grad-CAM Sample Heatmaps

---

## 🖥️ API Endpoints

| Method | Endpoint       | Description                          |
|--------|---------------|--------------------------------------|
| POST   | `/predict`    | Upload MRI + clinical data → prediction |
| GET    | `/health`     | Health check                         |
| GET    | `/model-info` | Model metadata                       |

---

## ⚠️ Disclaimer
This is a **research tool** and is **not intended for clinical diagnosis**. Always consult a qualified medical professional for health decisions.
