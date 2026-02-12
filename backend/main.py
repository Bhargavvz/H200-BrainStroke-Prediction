"""
Brain Stroke Prediction API — FastAPI Backend
Serves the trained Hybrid Deep Learning model for Brain MRI stroke prediction.

Endpoints:
  POST /predict     — Upload MRI image + clinical data → prediction + Grad-CAM
  GET  /health      — Health check
  GET  /model-info  — Model metadata
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    PredictionResponse, HealthResponse, ModelInfoResponse,
)
from model_service import prediction_service

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from config import IMG_SIZE, CLASS_NAMES, CLINICAL_FEATURES


# ── Lifespan (startup/shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    prediction_service.load_models()
    yield
    print("🔴 Shutting down model service.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🧠 Brain Stroke Prediction API",
    description=(
        "Hybrid Deep Learning system for Brain Stroke detection from MRI scans. "
        "Combines EfficientNet-B4 (image) + Clinical DNN (tabular) with late fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.is_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    return ModelInfoResponse(
        model_name="Brain Stroke Hybrid Predictor",
        architecture="EfficientNet-B4 + Clinical DNN (Late Fusion)",
        num_classes=len(CLASS_NAMES),
        class_names=prediction_service.class_names,
        image_size=IMG_SIZE,
        clinical_features=CLINICAL_FEATURES,
        fusion_strategy="Late Fusion (Feature Concatenation → FC Classifier)",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    mri_image: UploadFile = File(..., description="Brain MRI image (JPEG/PNG)"),
    clinical_data: str = Form(
        default="{}",
        description='JSON string of clinical data, e.g. {"age":67,"gender":"Male",...}',
    ),
):
    """
    Predict brain stroke from MRI image and optional clinical data.

    Accepts:
      - mri_image: MRI brain scan (JPEG or PNG)
      - clinical_data: JSON string with clinical features

    Returns:
      - Prediction label, stroke type, confidence scores
      - Grad-CAM heatmap (base64 PNG)
    """
    # Validate file type
    if mri_image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {mri_image.content_type}. "
                   "Only JPEG and PNG images are accepted.",
        )

    # Read image bytes
    image_bytes = await mri_image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    if len(image_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="Image file too large (max 50MB).")

    # Parse clinical data
    try:
        clinical = json.loads(clinical_data)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid clinical_data JSON format.",
        )

    # Run prediction
    try:
        result = prediction_service.predict(image_bytes, clinical)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    return PredictionResponse(**result)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
