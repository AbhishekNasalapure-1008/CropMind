"""
CropMind — Flask Application
─────────────────────────────────────────────────────────────────────────────
SETUP & RUN INSTRUCTIONS:
    1. pip install -r requirements.txt
    2. python model/train_model.py        ← First time only (trains the CNN)
    3. python app.py
    4. Open http://localhost:5000

NOTE: If model/npk_model.h5 does not exist, the app runs in DEMO MODE
      and returns realistic mock data so the frontend is always demonstrable.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import tempfile
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── Path setup so utils/ and model/ modules are importable ──────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from utils.image_processing import (
    preprocess_image,
    segment_leaf,
    filter_disease_vs_deficiency,
    generate_gradcam,
)
from utils.context_engine import adjust_for_context
from utils.recommendation import get_recommendation
from model.model_inference import load_model, predict, determine_severity, get_primary_deficiency

# ── App Init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load model once at startup
_model = load_model()
DEMO_MODE = _model is None

print(f"[CropMind] Running in {'DEMO' if DEMO_MODE else 'LIVE'} mode.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _demo_heatmap(preprocessed_image) -> str:
    """
    Generate a demo Grad-CAM-style heatmap overlay on the real preprocessed image.
    Uses a centre-weighted attention mask since we have no real activation maps.
    """
    import cv2, base64
    import numpy as np
    from io import BytesIO
    from PIL import Image

    img = (preprocessed_image * 255).astype(np.uint8)
    h, w = img.shape[:2]

    # Centre-weighted heat mask — higher attention in the middle
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / (w * 0.45)) ** 2 + ((Y - cy) / (h * 0.45)) ** 2)
    heat = np.clip(1 - dist, 0, 1)

    # Add slight randomness based on image content (bright leaf areas)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    heat = heat * (0.6 + 0.4 * gray)
    heat = (heat / heat.max() * 255).astype(np.uint8)

    colormap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.55, colormap_rgb, 0.45, 0)

    pil_img = Image.fromarray(overlay)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")



# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "model_loaded": not DEMO_MODE,
        "version": "1.0.0"
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    # ── Validate uploaded file ───────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload a JPG or PNG image."
        }), 415

    # ── Read context form fields ─────────────────────────────────────────────
    crop_type    = request.form.get("crop_type",    "other").strip().lower()
    soil_type    = request.form.get("soil_type",    "loamy").strip().lower()
    climate_zone = request.form.get("climate_zone", "temperate").strip().lower()
    growth_stage = request.form.get("growth_stage", "vegetative").strip().lower()
    notes        = request.form.get("notes",        "").strip()

    # ── Save temp file ───────────────────────────────────────────────────────
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()

        # ── Full pipeline (always runs — demo predict used when no model) ──────
        # 1. Preprocess
        preprocessed = preprocess_image(tmp.name)

        # 2. Segment leaf
        segmented = segment_leaf(preprocessed)

        # 3. Disease vs deficiency filter
        disease_check = filter_disease_vs_deficiency(segmented)

        # 4. Model inference (falls back to _demo_predict when model is None)
        raw_predictions = predict(preprocessed)

        # 5. Context adjustment
        adjusted = adjust_for_context(
            raw_predictions, crop_type, soil_type, climate_zone, growth_stage
        )

        # 6. Determine severity & primary deficiency
        # NOTE: primary is derived from RAW predictions so agronomic context
        # adjustments cannot flip which nutrient the image actually indicates.
        severity = determine_severity(adjusted)
        primary  = get_primary_deficiency(raw_predictions)
        confidence = round(
            max(adjusted.values()) * 0.7 + min(1.0, sum(adjusted.values()) * 0.15) * 0.3,
            3
        )

        # 7. Heatmap — Grad-CAM if real model, else demo heatmap overlay
        if DEMO_MODE:
            heatmap_b64 = _demo_heatmap(preprocessed)
        else:
            class_map = {"nitrogen": 0, "phosphorus": 1, "potassium": 2}
            class_idx = class_map.get(primary, 0)
            heatmap_b64 = generate_gradcam(_model, preprocessed, class_idx)

        # 8. Recommendations
        rec_package = get_recommendation(adjusted, severity, soil_type, crop_type)

        # ── Build response ───────────────────────────────────────────────────
        return jsonify({
            "success": True,
            "demo_mode": DEMO_MODE,
            "deficiencies": adjusted,
            "primary_deficiency": primary,
            "severity": severity,
            "confidence": confidence,
            "heatmap_base64": heatmap_b64,
            "recommendations": rec_package["recommendations"],
            "agronomic_insight": rec_package["agronomic_insight"],
            "symptoms_matched": rec_package["symptoms_matched"],
            "preventive_tips": rec_package["preventive_tips"],
            "disease_check": disease_check,
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "An internal server error occurred. Please try again."
        }), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
