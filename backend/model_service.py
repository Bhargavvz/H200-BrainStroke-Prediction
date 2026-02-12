"""
Model Service — Model Loading & Inference for Brain Stroke Prediction API.
Loads trained HybridFusionModel + ImageOnlyModel (for Grad-CAM) on startup.
Handles image preprocessing, clinical feature encoding, and prediction.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

# Add ML directory to path for model imports
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, ML_DIR)

from config import (
    DEVICE, MODEL_DIR, IMG_SIZE, NUM_CLASSES,
    CLASS_NAMES, CLINICAL_FEATURES, IMAGENET_MEAN, IMAGENET_STD,
    USE_AMP, AMP_DTYPE,
)
from models import HybridFusionModel, ImageOnlyModel
from preprocess import get_inference_transforms
from grad_cam import GradCAM, overlay_heatmap, heatmap_to_base64


class StrokePredictionService:
    """
    Manages model loading, preprocessing, and inference.
    """

    def __init__(self):
        self.hybrid_model = None
        self.image_model = None
        self.transform = None
        self.class_names = CLASS_NAMES
        self.is_loaded = False
        self.grad_cam = None

    def load_models(self):
        """Load trained models from disk."""
        print("🔄 Loading models...")

        # Load hybrid model
        hybrid_path = os.path.join(MODEL_DIR, "hybrid_model_best.pth")
        if os.path.exists(hybrid_path):
            self.hybrid_model = HybridFusionModel(pretrained=False).to(DEVICE)
            state_dict = torch.load(hybrid_path, map_location=DEVICE, weights_only=True)
            self.hybrid_model.load_state_dict(state_dict)
            self.hybrid_model.eval()
            print(f"  ✅ Hybrid model loaded from {hybrid_path}")
        else:
            print(f"  ⚠️  Hybrid model not found at {hybrid_path}")
            print("     Creating untrained model for demo mode...")
            self.hybrid_model = HybridFusionModel(pretrained=True).to(DEVICE)
            self.hybrid_model.eval()

        # Load image-only model (for Grad-CAM)
        img_path = os.path.join(MODEL_DIR, "image_only_model_best.pth")
        if os.path.exists(img_path):
            self.image_model = ImageOnlyModel(pretrained=False).to(DEVICE)
            state_dict = torch.load(img_path, map_location=DEVICE, weights_only=True)
            self.image_model.load_state_dict(state_dict)
            self.image_model.eval()
            self.grad_cam = GradCAM(self.image_model)
            print(f"  ✅ Image-only model loaded from {img_path}")
        else:
            print(f"  ⚠️  Image-only model not found, Grad-CAM will be disabled")

        # Load class names from training results if available
        results_path = os.path.join(MODEL_DIR, "test_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
                if "class_names" in results:
                    self.class_names = results["class_names"]

        self.transform = get_inference_transforms()
        self.is_loaded = True
        print(f"  ✅ Classes: {self.class_names}")
        print("🟢 Model service ready!")

    def preprocess_image(self, image_bytes: bytes) -> tuple:
        """
        Preprocess uploaded MRI image.
        Returns (preprocessed_tensor, original_pil_image)
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_image = image.copy()

        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)  # (1, 3, H, W)
        return tensor, original_image

    def encode_clinical_features(self, clinical_data: dict) -> torch.Tensor:
        """
        Encode clinical features into a tensor.
        Maps categorical features to numeric values.
        """
        features = []

        # Age (normalized 0-1 roughly)
        features.append(clinical_data.get("age", 50) / 100.0)

        # Gender (0=Female, 1=Male, 0.5=Other)
        gender = clinical_data.get("gender", "Other").lower()
        gender_map = {"male": 1.0, "female": 0.0, "other": 0.5}
        features.append(gender_map.get(gender, 0.5))

        # Hypertension (0/1)
        features.append(float(clinical_data.get("hypertension", 0)))

        # Heart disease (0/1)
        features.append(float(clinical_data.get("heart_disease", 0)))

        # Glucose (normalized)
        features.append(clinical_data.get("avg_glucose_level", 100) / 300.0)

        # BMI (normalized)
        features.append(clinical_data.get("bmi", 25) / 50.0)

        # Smoking status
        smoking = clinical_data.get("smoking_status", "unknown").lower()
        smoking_map = {
            "never_smoked": 0.0, "formerly_smoked": 0.5,
            "smokes": 1.0, "unknown": 0.25,
        }
        features.append(smoking_map.get(smoking, 0.25))

        # Cholesterol
        cholesterol = clinical_data.get("cholesterol", "normal").lower()
        chol_map = {"low": 0.0, "normal": 0.5, "high": 1.0}
        features.append(chol_map.get(cholesterol, 0.5))

        tensor = torch.FloatTensor([features])  # (1, 8)
        return tensor

    @torch.no_grad()
    def predict(self, image_bytes: bytes, clinical_data: dict = None) -> dict:
        """
        Run full prediction pipeline.

        Args:
            image_bytes: raw bytes of uploaded MRI image
            clinical_data: dict of clinical features (optional)

        Returns:
            dict with prediction, confidence, Grad-CAM, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Preprocess
        img_tensor, original_image = self.preprocess_image(image_bytes)
        img_tensor = img_tensor.to(DEVICE)

        # Clinical features
        if clinical_data:
            clinical_tensor = self.encode_clinical_features(clinical_data).to(DEVICE)
        else:
            clinical_tensor = torch.zeros(1, len(CLINICAL_FEATURES), device=DEVICE)

        # Inference
        if USE_AMP:
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                logits = self.hybrid_model(img_tensor, clinical_tensor)
        else:
            logits = self.hybrid_model(img_tensor, clinical_tensor)

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Confidence scores dict
        confidence_scores = {
            name: float(probs[i]) for i, name in enumerate(self.class_names)
        }

        # Stroke type
        if "ischemi" in pred_label.lower():
            stroke_type = "Ischemic"
        elif "hemorrh" in pred_label.lower() or "bleed" in pred_label.lower():
            stroke_type = "Hemorrhagic"
        else:
            stroke_type = "None"

        # Risk level
        if stroke_type != "None" and confidence > 0.7:
            risk_level = "High"
        elif stroke_type != "None" and confidence > 0.4:
            risk_level = "Medium"
        elif stroke_type != "None":
            risk_level = "Low"
        else:
            risk_level = "Low"

        # Grad-CAM
        grad_cam_b64 = None
        if self.image_model is not None and self.grad_cam is not None:
            try:
                heatmap, _, _ = self.grad_cam.generate(img_tensor, target_class=pred_idx)
                overlay = overlay_heatmap(original_image, heatmap, alpha=0.5)
                grad_cam_b64 = heatmap_to_base64(overlay)
            except Exception as e:
                print(f"  ⚠️  Grad-CAM generation failed: {e}")

        return {
            "prediction": pred_label,
            "stroke_type": stroke_type,
            "confidence": confidence,
            "confidence_scores": confidence_scores,
            "risk_level": risk_level,
            "grad_cam_image": grad_cam_b64,
        }


# Singleton instance
prediction_service = StrokePredictionService()
