"""
Pydantic Schemas for Brain Stroke Prediction API
Request validation & response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class ClinicalInput(BaseModel):
    """Clinical / tabular patient data input."""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Male / Female / Other")
    hypertension: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    avg_glucose_level: float = Field(..., ge=0, le=500, description="Average glucose level mg/dL")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    smoking_status: str = Field(..., description="never_smoked / formerly_smoked / smokes / unknown")
    cholesterol: str = Field(..., description="normal / high / low")


class PredictionResponse(BaseModel):
    """Prediction API response."""
    prediction: str = Field(..., description="Predicted class label")
    stroke_type: Optional[str] = Field(None, description="Ischemic / Hemorrhagic / None")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    confidence_scores: Dict[str, float] = Field(..., description="Per-class confidence scores")
    risk_level: str = Field(..., description="Low / Medium / High")
    grad_cam_image: Optional[str] = Field(None, description="Base64-encoded Grad-CAM heatmap PNG")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    architecture: str
    num_classes: int
    class_names: list
    image_size: int
    clinical_features: list
    fusion_strategy: str
