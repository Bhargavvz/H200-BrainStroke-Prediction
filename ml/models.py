"""
Model Architectures for Brain Stroke Prediction
  1. StrokeImageModel  — EfficientNet-B4 backbone (pretrained on ImageNet)
  2. ClinicalDNN       — 4-layer MLP for clinical/tabular features
  3. HybridFusionModel — Late fusion of image + clinical features
"""
import torch
import torch.nn as nn
import timm

from config import (
    NUM_CLASSES, NUM_CLINICAL_FEATURES,
    IMAGE_FEATURE_DIM, CLINICAL_FEATURE_DIM,
    FUSION_HIDDEN_DIM, DROPOUT_RATE, DROPOUT_RATE_FC,
)


class StrokeImageModel(nn.Module):
    """
    EfficientNet-B4 backbone for MRI image feature extraction.

    Why EfficientNet-B4:
      - Best accuracy-to-parameter ratio via compound scaling
      - Captures multi-scale features critical for stroke lesion detection
      - Pretrained on ImageNet — excellent transfer to medical images
      - Moderate size (19M params) — trains fast even on smaller datasets

    Input:  (B, 3, 380, 380) image tensor
    Output: (B, IMAGE_FEATURE_DIM) feature vector
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,           # Remove classifier → gives pooled features
            global_pool="avg",
        )
        backbone_out = self.backbone.num_features  # 1792 for EfficientNet-B4

        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE_FC),
            nn.Linear(512, IMAGE_FEATURE_DIM),
            nn.BatchNorm1d(IMAGE_FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)             # (B, 1792)
        features = self.feature_head(features)  # (B, 256)
        return features


class ClinicalDNN(nn.Module):
    """
    4-layer MLP for clinical / tabular features.

    Input features: age, gender, hypertension, heart_disease,
                    avg_glucose_level, bmi, smoking_status, cholesterol
    Input:  (B, NUM_CLINICAL_FEATURES) tensor
    Output: (B, CLINICAL_FEATURE_DIM) feature vector
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_CLINICAL_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(32, CLINICAL_FEATURE_DIM),
            nn.BatchNorm1d(CLINICAL_FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, 16)


class HybridFusionModel(nn.Module):
    """
    Late Fusion: concatenate image features + clinical features,
    then pass through a fusion classification head.

    Fusion Strategy — Late Fusion:
      - Image and clinical data have fundamentally different feature spaces
      - Late fusion allows each branch to learn domain-specific representations
        before combining them
      - Concatenation preserves all information from both branches
      - Fusion head learns optimal weighting automatically

    Input:  (image: (B,3,380,380), clinical: (B,8))
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.image_model = StrokeImageModel(pretrained=pretrained)
        self.clinical_model = ClinicalDNN()

        fusion_in = IMAGE_FEATURE_DIM + CLINICAL_FEATURE_DIM  # 256 + 16 = 272

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE_FC),
            nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, image, clinical):
        img_feat = self.image_model(image)         # (B, 256)
        cli_feat = self.clinical_model(clinical)   # (B, 16)
        fused = torch.cat([img_feat, cli_feat], dim=1)  # (B, 272)
        logits = self.fusion_head(fused)           # (B, 3)
        return logits

    def get_image_features(self, image):
        """Extract image features only (for Grad-CAM)."""
        return self.image_model(image)


class ImageOnlyModel(nn.Module):
    """
    Standalone image classifier for Grad-CAM and image-only inference.
    Wraps StrokeImageModel with a classification head.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.image_model = StrokeImageModel(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE_FC),
            nn.Linear(IMAGE_FEATURE_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        features = self.image_model(x)
        return self.classifier(features)

    @property
    def backbone(self):
        return self.image_model.backbone
