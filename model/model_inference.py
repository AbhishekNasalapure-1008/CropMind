"""
CropMind — Model Inference Module
Loads the saved NPK model and runs predictions on preprocessed images.
"""

import os
import sys
import numpy as np

# ─────────────────────────────────────────────
# Model path resolution
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(_BASE_DIR, "model", "npk_model.h5")
CLASSES_PATH = os.path.join(_BASE_DIR, "model", "label_classes.npy")

_model = None  # Singleton


def load_model():
    """Load and cache the Keras model. Called once at Flask startup."""
    global _model

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}. Using DEMO mode.")
        return None

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def _demo_predict(image: np.ndarray) -> dict:
    """
    Fallback demo predictor — uses real image color analysis to distinguish
    between Nitrogen (yellowing), Phosphorus (purple tint), and Potassium
    (edge browning / marginal yellowing) deficiency patterns.

    Leaf mask strictly requires green dominance over BOTH R and B so that
    neutral/gray backgrounds are excluded from all channel statistics.
    """
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)

    # ── Strict leaf mask: green must dominate both red AND blue ─────────────
    # This excludes gray/beige backgrounds where r≈g≈b (previously inflating P)
    leaf_mask = (g > 0.10) & (g > r * 0.85) & (g > b * 0.85)
    leaf_area = float(np.sum(leaf_mask))
    if leaf_area < 400:
        # Very little green detected — fall back to green-channel dominance only
        leaf_mask = (g > 0.10) & (g > r * 0.65)
        leaf_area = float(np.sum(leaf_mask))
    if leaf_area < 200:
        leaf_mask = np.ones_like(g, dtype=bool)
        leaf_area = float(leaf_mask.sum())

    r_l = r[leaf_mask]
    g_l = g[leaf_mask]
    b_l = b[leaf_mask]

    mean_r = float(np.mean(r_l))
    mean_g = float(np.mean(g_l))
    mean_b = float(np.mean(b_l))

    # ── Signal 1: NITROGEN — yellowing (R/G ratio high, yellow pixels) ──────
    rg_ratio = mean_r / (mean_g + 1e-6)
    # Classic yellow: R↑ G↑ B↓
    yellow_px = float(np.sum((r_l > 0.45) & (g_l > 0.35) & (b_l < 0.30)))
    yellow_fraction = yellow_px / (leaf_area + 1e-6)
    # Pale/bleached: channels converge at mid-range
    pale_px = float(np.sum((np.abs(r_l - g_l) < 0.12) & (g_l > 0.20) & (g_l < 0.65)))
    pale_fraction = pale_px / (leaf_area + 1e-6)

    nitrogen = float(np.clip(
        (rg_ratio - 0.55) * 1.3 +
        yellow_fraction * 2.0 +
        pale_fraction * 0.5 -
        mean_g * 0.5,
        0.0, 1.0
    ))

    # ── Signal 2: PHOSPHORUS — true purple/reddish tint (B+R high, G low) ──
    # Exclude near-gray pixels (|R-G| and |B-G| both < 0.10) to avoid
    # counting neutral backgrounds as purple tissue
    non_gray = (np.abs(r_l - g_l) > 0.10) | (np.abs(b_l - g_l) > 0.10)
    purple_px = float(np.sum(
        non_gray & (r_l > 0.20) & (b_l > 0.18) &
        ((r_l + b_l) > g_l * 1.5)          # R+B clearly exceeds G
    ))
    purple_fraction = purple_px / (leaf_area + 1e-6)
    dark_px = float(np.sum((r_l + g_l + b_l) < 0.50))
    dark_fraction = dark_px / (leaf_area + 1e-6)
    bg_ratio = mean_b / (mean_g + 1e-6)

    phosphorus = float(np.clip(
        purple_fraction * 3.0 +
        dark_fraction * 0.7 +
        (bg_ratio - 0.40) * 1.0 -
        mean_g * 0.3,
        0.0, 1.0
    ))

    # ── Signal 3: POTASSIUM — marginal browning AND yellow margins ──────────
    h, w = image.shape[:2]
    margin = max(int(h * 0.22), 8)

    def _edge_band(arr):
        return np.concatenate([
            arr[:margin, :].ravel(), arr[-margin:, :].ravel(),
            arr[:, :margin].ravel(), arr[:, -margin:].ravel()
        ])

    edge_r = _edge_band(r)
    edge_g = _edge_band(g)
    edge_b = _edge_band(b)
    edge_total = float(len(edge_r)) + 1e-6

    # Brown scorch at margins: R dominant, B low
    brown_edge = float(np.sum(
        (edge_r > 0.32) & (edge_r > edge_g * 1.20) & (edge_b < 0.28)
    ))
    # Early K: YELLOW at margins (interveinal chlorosis starts at edges)
    yellow_edge = float(np.sum(
        (edge_r > 0.40) & (edge_g > 0.30) & (edge_b < 0.28) &
        (edge_r > edge_g * 0.80)           # R not too far above G (yellow not brown)
    ))
    # Scorch pixels anywhere on leaf
    scorch_px = float(np.sum((r_l > 0.55) & (r_l > g_l * 1.55) & (b_l < 0.28)))

    potassium = float(np.clip(
        (brown_edge / edge_total) * 3.5 +
        (yellow_edge / edge_total) * 2.0 +  # yellow margins = early K def
        (scorch_px / (leaf_area + 1e-6)) * 1.8 +
        (mean_r - mean_g) * 0.6 -
        0.05,
        0.0, 1.0
    ))

    return {
        "nitrogen":   round(nitrogen,   3),
        "phosphorus": round(phosphorus, 3),
        "potassium":  round(potassium,  3),
    }



def predict(preprocessed_image: np.ndarray) -> dict:
    """
    Run NPK deficiency inference.

    Args:
        preprocessed_image: float32 numpy array of shape (224, 224, 3), values [0,1].

    Returns:
        dict: {"nitrogen": float, "phosphorus": float, "potassium": float}
              all values clamped to [0.0, 1.0].
    """
    global _model

    if _model is None:
        return _demo_predict(preprocessed_image)

    try:
        import tensorflow as tf
        batch = tf.convert_to_tensor(preprocessed_image[np.newaxis, ...])
        preds = _model.predict(batch, verbose=0)
        preds = preds[0]  # shape (3,)

        return {
            "nitrogen":   float(round(float(np.clip(preds[0], 0.0, 1.0)), 3)),
            "phosphorus": float(round(float(np.clip(preds[1], 0.0, 1.0)), 3)),
            "potassium":  float(round(float(np.clip(preds[2], 0.0, 1.0)), 3)),
        }
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}. Falling back to demo mode.")
        return _demo_predict(preprocessed_image)


def determine_severity(deficiencies: dict) -> str:
    """
    Determine overall severity from the highest deficiency score.
    Returns: 'none' | 'low' | 'moderate' | 'severe'
    """
    max_score = max(deficiencies.values())
    if max_score >= 0.70:
        return "severe"
    elif max_score >= 0.45:
        return "moderate"
    elif max_score >= 0.25:
        return "low"
    return "none"


def get_primary_deficiency(deficiencies: dict) -> str:
    """Return the nutrient with the highest deficiency score."""
    return max(deficiencies, key=deficiencies.get)
