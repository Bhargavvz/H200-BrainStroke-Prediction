"""
Model Architectures for Brain Stroke Prediction
  1. StrokeImageModel  — EfficientNet-B4 backbone (pretrained on ImageNet)
  2. ClinicalDNN       — 4-layer MLP for clinical/tabular features
  3. HybridFusionModel — Late fusion of image + clinical features
  4. ImageOnlyModel    — For standalone inference & Grad-CAM

Freeze/Unfreeze API:
  model.freeze_backbone()   — Phase 1: train only heads
  model.unfreeze_backbone() — Phase 2: fine-tune everything (BN stays frozen)
  model.get_param_groups()  — Returns discriminative LR groups

IMPORTANT: Backbone BatchNorm layers are ALWAYS kept in eval mode
to preserve ImageNet running statistics. This prevents the val loss
explosion that occurs when BN switches to batch statistics on a
small/different-domain dataset.
"""
import torch
import torch.nn as nn
import timm

from config import (
    NUM_CLASSES, NUM_CLINICAL_FEATURES,
    IMAGE_FEATURE_DIM, CLINICAL_FEATURE_DIM,
    FUSION_HIDDEN_DIM, DROPOUT_RATE, DROPOUT_RATE_FC,
)


def _freeze_bn(module):
    """Recursively set all BatchNorm layers to eval mode."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()
            # Prevent running stats from updating
            m.track_running_stats = False


def _unfreeze_bn(module):
    """Restore BatchNorm tracking (only for head BN, not backbone)."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.track_running_stats = True


class StrokeImageModel(nn.Module):
    """
    EfficientNet-B4 backbone for MRI image feature extraction.

    Input:  (B, 3, 380, 380) image tensor
    Output: (B, IMAGE_FEATURE_DIM) feature vector
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_out = self.backbone.num_features  # 1792

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

        self._backbone_frozen = False

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_head(features)
        return features

    def train(self, mode=True):
        """
        Override train() to ALWAYS keep backbone BN in eval mode.
        This preserves ImageNet running statistics during fine-tuning.
        """
        super().train(mode)
        if mode and self._backbone_frozen:
            # When backbone is frozen, keep entire backbone in eval
            self.backbone.eval()
        elif mode:
            # When backbone is unfrozen, keep ONLY BN in eval
            _freeze_bn(self.backbone)
        return self

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1)."""
        self._backbone_frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Unfreeze backbone params but keep BN frozen (Phase 2)."""
        self._backbone_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
        # BN stays in eval via the train() override above


class ClinicalDNN(nn.Module):
    """
    4-layer MLP for clinical / tabular features.
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
        return self.net(x)


class HybridFusionModel(nn.Module):
    """
    Late Fusion: image features + clinical features → classification.

    Input:  (image: (B,3,380,380), clinical: (B,8))
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.image_model = StrokeImageModel(pretrained=pretrained)
        self.clinical_model = ClinicalDNN()

        fusion_in = IMAGE_FEATURE_DIM + CLINICAL_FEATURE_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE_FC),
            nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, image, clinical):
        img_feat = self.image_model(image)
        cli_feat = self.clinical_model(clinical)
        fused = torch.cat([img_feat, cli_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def get_image_features(self, image):
        return self.image_model(image)

    def freeze_backbone(self):
        self.image_model.freeze_backbone()
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f"   🔒 Backbone frozen: {frozen}/{total} parameter groups frozen")

    def unfreeze_backbone(self):
        self.image_model.unfreeze_backbone()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   🔓 Backbone unfrozen: {trainable:,} trainable params (BN stays frozen)")

    def get_param_groups(self, lr_backbone, lr_head):
        """Discriminative LR: backbone gets lower LR to preserve pretrained features."""
        backbone_params = [p for p in self.image_model.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.image_model.feature_head.parameters()) +
            list(self.clinical_model.parameters()) +
            list(self.fusion_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "name": "head"},
        ]


class ImageOnlyModel(nn.Module):
    """Standalone image classifier for Grad-CAM and image-only inference."""

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

    def freeze_backbone(self):
        self.image_model.freeze_backbone()

    def unfreeze_backbone(self):
        self.image_model.unfreeze_backbone()

    def get_param_groups(self, lr_backbone, lr_head):
        backbone_params = [p for p in self.image_model.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.image_model.feature_head.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "name": "head"},
        ]
