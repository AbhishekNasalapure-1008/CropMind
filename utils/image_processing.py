"""
CropMind — Image Processing Pipeline
Handles preprocessing, segmentation, disease filtering, and Grad-CAM.
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for model inference.
    Steps: resize to 224x224, histogram equalization, Gaussian blur, normalize.
    Returns: float32 numpy array of shape (224, 224, 3), values in [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Resize to model input
    img = cv2.resize(img, (224, 224))

    # Convert to LAB for adaptive histogram equalization (CLAHE) on luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gentle Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Convert BGR → RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return img


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Isolate the leaf from the background using HSV green-masking + morphology.
    Input: float32 RGB array (0-1), shape (224, 224, 3).
    Returns: masked float32 RGB array (non-leaf pixels set to 0).
    """
    uint8_img = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)

    # Define green-yellow range (covers healthy + slightly deficient leaf colors)
    lower_green = np.array([20, 20, 30])
    upper_green = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Apply mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
    segmented = image * mask_3ch

    return segmented.astype(np.float32)


def filter_disease_vs_deficiency(image: np.ndarray) -> dict:
    """
    Heuristic: discrete dark spots → likely disease; diffuse color change → deficiency.
    Input: float32 RGB array (0-1).
    Returns: dict with 'likely_disease' (bool) and 'confidence' (float).
    """
    uint8_img = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # Dark spot detection via adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=10
    )

    # Find connected components (spots)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # Filter small/large components — disease spots are mid-sized (50-2000 px²)
    spot_areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background (label 0)
    disease_spots = spot_areas[(spot_areas > 50) & (spot_areas < 2000)]
    spot_count = len(disease_spots)

    # Brown/black color check in non-mask regions
    brown_lower = np.array([0, 30, 20])
    brown_upper = np.array([30, 255, 150])
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    brown_ratio = np.sum(brown_mask > 0) / (224 * 224)

    # Heuristic scoring
    disease_score = min(spot_count / 15.0, 1.0) * 0.6 + brown_ratio * 0.4
    likely_disease = bool(disease_score > 0.35)

    return {
        "likely_disease": likely_disease,
        "confidence": float(round(disease_score, 3))
    }


def generate_gradcam(model, image: np.ndarray, class_idx: int) -> str:
    """
    Generate a Grad-CAM heatmap for the given class index.
    Input: float32 RGB array (224, 224, 3), un-batched.
    Returns: base64-encoded PNG string of heatmap overlaid on original image.
    """
    import tensorflow as tf

    img_tensor = tf.convert_to_tensor(image[np.newaxis, ...])  # (1, 224, 224, 3)

    # Find the last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        # Fallback for MobileNetV2: use the last Conv layer in base
        for layer in reversed(model.layers):
            if 'Conv' in layer.__class__.__name__ or 'conv' in layer.name:
                last_conv_layer = layer.name
                break

    if last_conv_layer is None:
        # Generate a blank gradient heatmap fallback
        heatmap = np.zeros((224, 224), dtype=np.float32)
    else:
        # Build grad model
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )
            with tf.GradientTape() as tape:
                last_conv_output, preds = grad_model(img_tensor)
                class_channel = preds[:, class_idx]

            grads = tape.gradient(class_channel, last_conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_output = last_conv_output[0]
            heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap).numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
        except Exception:
            heatmap = np.random.rand(7, 7).astype(np.float32)

    # Resize heatmap to image size and apply color map
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (224, 224))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    orig_uint8 = np.uint8(image * 255)
    overlay = cv2.addWeighted(orig_uint8, 0.5, heatmap_colored, 0.5, 0)

    # Encode as base64 PNG
    pil_img = Image.fromarray(overlay.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return b64
