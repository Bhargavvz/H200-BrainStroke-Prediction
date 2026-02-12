"""
Grad-CAM Explainability for Brain Stroke MRI
Generates class-activation heatmaps overlaid on original MRI images,
showing which regions the model focuses on for its prediction.
"""
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from config import DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """
    Grad-CAM for EfficientNet-B4 backbone.
    Hooks into the last convolutional layer to capture gradients
    and activations, then produces a class-specific heatmap.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: ImageOnlyModel or the image backbone
            target_layer: nn.Module — the conv layer to hook.
                          If None, auto-selects the last conv block.
        """
        self.model = model
        self.model.eval()

        # Auto-detect target layer (last conv block in EfficientNet)
        if target_layer is None:
            # For ImageOnlyModel → backbone is EfficientNet
            if hasattr(model, "backbone"):
                backbone = model.backbone
            elif hasattr(model, "image_model"):
                backbone = model.image_model.backbone
            else:
                backbone = model

            # EfficientNet's last feature-extraction block
            if hasattr(backbone, "conv_head"):
                target_layer = backbone.conv_head
            elif hasattr(backbone, "blocks"):
                target_layer = backbone.blocks[-1]
            else:
                # Fallback: last child module that is Conv2d
                for m in reversed(list(backbone.modules())):
                    if isinstance(m, torch.nn.Conv2d):
                        target_layer = m
                        break

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            target_class: int, class index. If None, uses argmax prediction.

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
            prediction: int, predicted class index
            confidence: float, softmax probability
        """
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]          # (C, h, w)
        activations = self.activations[0]      # (C, h, w)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))    # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (h, w)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        return cam, target_class, confidence


def overlay_heatmap(original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        heatmap: numpy array (H, W), values in [0, 1]
        alpha: float, blending factor
        colormap: OpenCV colormap

    Returns:
        overlay: numpy array (H, W, 3) BGR
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.convert("RGB"))

    # Resize original to match heatmap
    original_resized = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))

    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_resized)
    return overlay


def heatmap_to_base64(overlay_image):
    """Convert overlay image (numpy array) to base64 string for API response."""
    img = Image.fromarray(overlay_image)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_gradcam_base64(model, input_tensor, original_image, target_class=None, alpha=0.5):
    """
    Full pipeline: generate Grad-CAM and return base64-encoded overlay.

    Args:
        model: trained ImageOnlyModel
        input_tensor: (1, 3, H, W) preprocessed tensor
        original_image: PIL Image of the original MRI
        target_class: optional target class
        alpha: overlay opacity

    Returns:
        dict with keys: grad_cam_image (base64), prediction, confidence
    """
    cam = GradCAM(model)
    heatmap, pred_class, confidence = cam.generate(input_tensor, target_class)
    overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)
    b64 = heatmap_to_base64(overlay)

    return {
        "grad_cam_image": b64,
        "prediction": pred_class,
        "confidence": confidence,
    }
