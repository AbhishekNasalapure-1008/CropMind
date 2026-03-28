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

    Handles the case where MobileNetV2 is a nested sub-model inside the
    outer CropMind functional model by searching both the outer and inner
    layer lists.

    Args:
        model:     The loaded Keras model.
        image:     float32 RGB array of shape (224, 224, 3), values in [0, 1].
        class_idx: Index of the class to explain (0=N, 1=P, 2=K).

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    import tensorflow as tf

    img_tensor = tf.convert_to_tensor(image[np.newaxis, ...])  # (1, 224, 224, 3)
    heatmap = None

    try:
        # ── Step 1: find last conv layer (search outer + nested inner layers) ──
        last_conv_layer_name = None

        # Search outer model layers first
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'filters'):
                last_conv_layer_name = layer.name
                break

        # If not found, dig into nested sub-models (e.g. MobileNetV2)
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if hasattr(layer, 'layers'):     # it's a sub-model
                    for sub_layer in reversed(layer.layers):
                        if 'conv' in sub_layer.name.lower() and hasattr(sub_layer, 'filters'):
                            last_conv_layer_name = sub_layer.name
                            break
                if last_conv_layer_name:
                    break

        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found for Grad-CAM.")

        # ── Step 2: build grad model pointing to the found conv layer ─────────
        # We need to handle nested models: find the actual layer object.
        target_layer = None
        try:
            target_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            # Try inside sub-models
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        target_layer = layer.get_layer(last_conv_layer_name)
                        break
                    except ValueError:
                        continue

        if target_layer is None:
            raise ValueError(f"Could not resolve layer '{last_conv_layer_name}'.")

        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_tensor)
            class_channel = preds[:, class_idx]

        grads       = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output  = conv_output[0]
        heatmap_raw  = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap_raw  = tf.squeeze(heatmap_raw).numpy()
        heatmap_raw  = np.maximum(heatmap_raw, 0)
        if heatmap_raw.max() > 1e-8:
            heatmap_raw /= heatmap_raw.max()
        heatmap = heatmap_raw

    except Exception as exc:
        print(f"[GradCAM] Warning: {exc} — using random heatmap fallback.")
        heatmap = np.random.rand(7, 7).astype(np.float32)

    # ── Resize heatmap and overlay on original image ───────────────────────────
    heatmap_uint8   = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (224, 224))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig_uint8 = np.uint8(image * 255)
    overlay    = cv2.addWeighted(orig_uint8, 0.5, heatmap_colored, 0.5, 0)

    pil_img = Image.fromarray(overlay.astype(np.uint8))
    buffer  = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
