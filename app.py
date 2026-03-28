"""
CropMind — Flask Application
──────────────────────────────────────────────────────────────────────────────
SETUP & RUN:
    1.  pip install -r requirements.txt
    2.  python model/train_model.py        ← First-time model training
    3.  python app.py
    4.  Open http://localhost:5000

NOTE: If model/npk_model.h5 does not exist, the app runs in DEMO MODE
      and returns realistic results based on colour-channel analysis,
      so the frontend is always demonstrable without a trained model.

ENDPOINTS:
    GET  /                          → Main SPA
    GET  /api/health                → Health check + mode info
    POST /api/analyze               → Main image analysis pipeline
    POST /api/feedback              → Submit label correction (self-learning)
    GET  /api/feedback/stats        → Feedback dataset statistics
    POST /api/retrain               → Trigger fine-tuning on feedback data
    GET  /api/retrain/status        → Current retraining job status
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import uuid
import tempfile
import traceback
import shutil

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from utils.image_processing  import (
    preprocess_image,
    segment_leaf,
    filter_disease_vs_deficiency,
    generate_gradcam,
)
from utils.context_engine    import adjust_for_context
from utils.recommendation    import get_recommendation
from utils.feedback_manager  import (
    save_feedback_image,
    get_feedback_stats,
    get_retrain_status,
    retrain_in_background,
)
from model.model_inference   import (
    load_model,
    predict,
    determine_severity,
    get_primary_deficiency,
)

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH  = 10 * 1024 * 1024     # 10 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── Persistent temp directory for feedback images ─────────────────────────────
# Unlike tempfile.NamedTemporaryFile (auto-deleted), we keep the file alive
# until the user optionally submits feedback, then copy it to feedback_data/.
PENDING_DIR = os.path.join(_BASE_DIR, "pending_feedback")
os.makedirs(PENDING_DIR, exist_ok=True)

# ── Load model once at startup ────────────────────────────────────────────────
_model    = load_model()
DEMO_MODE = _model is None
print(f"[CropMind] Running in {'DEMO' if DEMO_MODE else 'LIVE'} mode.")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _demo_heatmap(preprocessed_image) -> str:
    """Centre-weighted attention heatmap overlay for demo mode (no real model needed)."""
    import cv2, base64
    import numpy as np
    from io import BytesIO
    from PIL import Image

    img    = (preprocessed_image * 255).astype(np.uint8)
    h, w   = img.shape[:2]
    cx, cy = w // 2, h // 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt(((X - cx) / (w * 0.45)) ** 2 + ((Y - cy) / (h * 0.45)) ** 2)
    heat   = np.clip(1 - dist, 0, 1)

    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    heat   = heat * (0.6 + 0.4 * gray)
    heat   = (heat / heat.max() * 255).astype("uint8")

    colormap     = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(img, 0.55, colormap_rgb, 0.45, 0)

    buf = BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _save_pending(tmp_path: str, prediction_id: str, ext: str) -> str:
    """
    Copy the analyzed temp image to pending_feedback/<prediction_id><ext>
    so it can be retrieved later if the user submits a feedback correction.
    Returns the destination path.
    """
    dest = os.path.join(PENDING_DIR, f"{prediction_id}{ext}")
    try:
        shutil.copy2(tmp_path, dest)
    except Exception as e:
        print(f"[WARNING] Could not save pending image: {e}")
        dest = ""
    return dest


def _cleanup_old_pending(max_age_seconds: int = 3600):
    """Remove pending images older than max_age_seconds (default 1 hour)."""
    import time
    now = time.time()
    for fname in os.listdir(PENDING_DIR):
        fpath = os.path.join(PENDING_DIR, fname)
        try:
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                os.unlink(fpath)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Core
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "demo_mode":    DEMO_MODE,
        "model_loaded": not DEMO_MODE,
        "version":      "2.0.0",
    })


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Analysis
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main image analysis endpoint.

    Form fields:
        image        (file)   — leaf image
        crop_type    (str)    — e.g. "rice"
        soil_type    (str)    — e.g. "loamy"
        climate_zone (str)    — e.g. "tropical"
        growth_stage (str)    — e.g. "vegetative"
        notes        (str)    — optional free text

    Returns JSON with:
        success, demo_mode, prediction_id,
        deficiencies, primary_deficiency, severity, confidence,
        heatmap_base64, recommendations, agronomic_insight,
        symptoms_matched, preventive_tips, disease_check
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload JPG, PNG or WebP."
        }), 415

    # ── Parse form context ────────────────────────────────────────────────────
    crop_type    = request.form.get("crop_type",    "other").strip().lower()
    soil_type    = request.form.get("soil_type",    "loamy").strip().lower()
    climate_zone = request.form.get("climate_zone", "temperate").strip().lower()
    growth_stage = request.form.get("growth_stage", "vegetative").strip().lower()

    ext        = "." + file.filename.rsplit(".", 1)[-1].lower()
    prediction_id = str(uuid.uuid4())      # unique ID for feedback traceability

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        file.save(tmp.name)
        tmp.close()

        _cleanup_old_pending()

        # ── Full analysis pipeline ────────────────────────────────────────────
        preprocessed   = preprocess_image(tmp.name)
        segmented      = segment_leaf(preprocessed)
        disease_check  = filter_disease_vs_deficiency(segmented)
        raw_predictions = predict(preprocessed)

        adjusted  = adjust_for_context(raw_predictions, crop_type, soil_type,
                                       climate_zone, growth_stage)
        severity  = determine_severity(adjusted)
        primary   = get_primary_deficiency(raw_predictions)   # always from RAW
        confidence = round(
            max(adjusted.values()) * 0.7 +
            min(1.0, sum(adjusted.values()) * 0.15) * 0.3,
            3,
        )

        # ── Heatmap ───────────────────────────────────────────────────────────
        if DEMO_MODE:
            heatmap_b64 = _demo_heatmap(preprocessed)
        else:
            class_map   = {"nitrogen": 0, "phosphorus": 1, "potassium": 2}
            class_idx   = class_map.get(primary, 0)
            heatmap_b64 = generate_gradcam(_model, preprocessed, class_idx)

        # ── Recommendations ───────────────────────────────────────────────────
        rec_pkg = get_recommendation(adjusted, severity, soil_type, crop_type)

        # ── Save image for potential feedback later ───────────────────────────
        _save_pending(tmp.name, prediction_id, ext)

        return jsonify({
            "success":           True,
            "demo_mode":         DEMO_MODE,
            "prediction_id":     prediction_id,
            "deficiencies":      adjusted,
            "primary_deficiency": primary,
            "severity":          severity,
            "confidence":        confidence,
            "heatmap_base64":    heatmap_b64,
            "recommendations":   rec_pkg["recommendations"],
            "agronomic_insight": rec_pkg["agronomic_insight"],
            "symptoms_matched":  rec_pkg["symptoms_matched"],
            "preventive_tips":   rec_pkg["preventive_tips"],
            "disease_check":     disease_check,
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error":   "An internal server error occurred. Please try again.",
        }), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Self-Learning Feedback
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Receive a user feedback correction and store the image in feedback_data/.

    JSON or form body:
        prediction_id  (str) — from the /api/analyze response
        correct_label  (str) — nitrogen | phosphorus | potassium | healthy
        predicted_label (str, optional) — model's original prediction
        confidence      (float, optional)

    The previously saved pending image is moved to feedback_data/<correct_label>/.
    """
    body = request.get_json(silent=True) or request.form

    prediction_id  = (body.get("prediction_id") or "").strip()
    correct_label  = (body.get("correct_label")  or "").strip().lower()
    predicted_label = (body.get("predicted_label") or "").strip().lower()

    try:
        confidence = float(body.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if not prediction_id:
        return jsonify({"success": False, "error": "prediction_id is required."}), 400
    if not correct_label:
        return jsonify({"success": False, "error": "correct_label is required."}), 400

    # Find the pending image for this prediction
    source_path = ""
    for fname in os.listdir(PENDING_DIR):
        if fname.startswith(prediction_id):
            source_path = os.path.join(PENDING_DIR, fname)
            break

    if not source_path:
        return jsonify({
            "success": False,
            "error":   "Prediction image not found. Images expire after 1 hour.",
        }), 404

    result = save_feedback_image(
        source_image_path=source_path,
        correct_label=correct_label,
        prediction_id=prediction_id,
        predicted_label=predicted_label,
        confidence=confidence,
    )

    if result["success"]:
        # Remove from pending after successful copy
        try:
            os.unlink(source_path)
        except Exception:
            pass
        stats = get_feedback_stats()
        result["feedback_stats"] = stats
        result["retrain_available"] = stats["ready_to_retrain"]

    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/api/feedback/stats", methods=["GET"])
def feedback_stats():
    """Return per-class feedback image counts and retraining readiness."""
    stats = get_feedback_stats()
    return jsonify({"success": True, **stats})


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES — Retraining
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/retrain", methods=["POST"])
def retrain():
    """
    Trigger model fine-tuning on accumulated feedback data.

    Retraining runs in a background thread so this endpoint returns immediately.
    Check /api/retrain/status for progress.

    Requirements:
        At least 10 feedback images must be stored (enforced by feedback_manager).
    """
    stats = get_feedback_stats()
    if not stats["ready_to_retrain"]:
        return jsonify({
            "success": False,
            "error":   (
                f"Not enough feedback data to retrain. "
                f"Have {stats['total']} samples; need at least 10."
            ),
            "feedback_stats": stats,
        }), 400

    result = retrain_in_background()
    status_code = 200 if result.get("queued") else 409   # 409 Conflict = already running
    return jsonify({"success": result.get("queued", False), **result}), status_code


@app.route("/api/retrain/status", methods=["GET"])
def retrain_status():
    """Return current background retraining job status."""
    return jsonify({"success": True, **get_retrain_status()})


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
