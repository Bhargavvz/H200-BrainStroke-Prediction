"""
Grad-CAM Explainability for Brain Stroke MRI
Generates class-activation heatmaps overlaid on original MRI images,
showing which regions the model focuses on for its prediction.

Uses PIL + matplotlib instead of OpenCV for maximum compatibility.
"""
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import DEVICE, IMG_SIZE


class GradCAM:
    """
    Grad-CAM for EfficientNet-B4 backbone.
    Hooks into the last convolutional layer to capture gradients
    and activations, then produces a class-specific heatmap.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Auto-detect target layer (last conv block in EfficientNet)
        if target_layer is None:
            if hasattr(model, "backbone"):
                backbone = model.backbone
            elif hasattr(model, "image_model"):
                backbone = model.image_model.backbone
            else:
                backbone = model

            if hasattr(backbone, "conv_head"):
                target_layer = backbone.conv_head
            elif hasattr(backbone, "blocks"):
                target_layer = backbone.blocks[-1]
            else:
                for m in reversed(list(backbone.modules())):
                    if isinstance(m, torch.nn.Conv2d):
                        target_layer = m
                        break

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            target_class: int, class index. If None, uses argmax.

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
            prediction: int, predicted class index
            confidence: float, softmax probability
        """
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients[0]       # (C, h, w)
        activations = self.activations[0]   # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam_map = (weights[:, None, None] * activations).sum(dim=0)
        cam_map = F.relu(cam_map)

        if cam_map.max() > 0:
            cam_map = cam_map / cam_map.max()

        cam_np = cam_map.cpu().numpy()

        # Resize using PIL (no cv2 needed)
        cam_pil = Image.fromarray(cam_np)
        cam_pil = cam_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heatmap = np.array(cam_pil)

        return heatmap, target_class, confidence


def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image using matplotlib colormap.

    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        heatmap: numpy array (H, W), values in [0, 1]
        alpha: float, blending factor

    Returns:
        overlay: numpy array (H, W, 3) RGB, uint8
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.convert("RGB"))

    # Resize original to match heatmap
    h, w = heatmap.shape
    orig_pil = Image.fromarray(original_image)
    orig_pil = orig_pil.resize((w, h), Image.BILINEAR)
    original_resized = np.array(orig_pil)

    # Apply JET colormap (same as cv2.COLORMAP_JET)
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

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

    Returns:
        dict with keys: grad_cam_image (base64), prediction, confidence
    """
    grad_cam = GradCAM(model)
    heatmap, pred_class, confidence = grad_cam.generate(input_tensor, target_class)
    overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)
    b64 = heatmap_to_base64(overlay)

    return {
        "grad_cam_image": b64,
        "prediction": pred_class,
        "confidence": confidence,
    }
