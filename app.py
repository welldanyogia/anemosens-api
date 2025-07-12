#!/usr/bin/env python3
"""
app.py — Flask service for anemia prediction with EfficientNet-based model.
Includes a hack to suppress Colab’s debugpy “get_shape” errors when run inside notebooks.
"""

# ─── Debugpy / Colab repr hack ──────────────────────────────────────────────────
# If running in Colab, patch google.colab._debugpy_repr.get_shape so it never raises.
try:
    import google.colab._debugpy_repr as _repr
    _orig_get_shape = _repr.get_shape

    def _safe_get_shape(obj):
        try:
            return _orig_get_shape(obj)
        except Exception:
            return None

    _repr.get_shape = _safe_get_shape
except ImportError:
    # Not in Colab; no patch needed.
    pass

# ─── Standard imports ──────────────────────────────────────────────────────────
import logging
import os
import base64

import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─── Configuration ──────────────────────────────────────────────────────────────
# Silence all logs below ERROR level
logging.getLogger().setLevel(logging.ERROR)

# Path to your trained model (update as needed)
MODEL_PATH = 'model_anemia.h5'

# Input image size for the model
IMG_SIZE = (224, 224)

# ─── Load the model ─────────────────────────────────────────────────────────────
# The semicolon suppresses Jupyter repr if accidentally loaded in a cell
model = tf.keras.models.load_model(MODEL_PATH, compile=False);

# ─── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

def prep_image_for_flask(img_data: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """
    Decode raw JPEG/PNG bytes, convert to RGB, resize to `img_size`, preprocess,
    and return a batch of shape (1, H, W, C).
    """
    # Convert to OpenCV image
    arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")

    # BGR->RGB, resize, and EfficientNet preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # —————— CASE 1: multipart/form-data ——————
        if request.content_type.startswith('multipart/'):
            # ambil file image
            img_file = request.files.get('image') or request.files.get('file')
            if not img_file:
                return jsonify({"error": "No image file provided"}), 400

            img_bytes = img_file.read()

            # metadata dari form-data
            age = float(request.form.get('age', 0))
            gender = request.form.get('gender', 'M').strip().upper()
        # —————— CASE 2: application/json ——————
        else:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON or multipart/form-data"}), 415
            data = request.get_json()
            img_b64 = data.get('image')
            if not img_b64:
                return jsonify({"error": "No image data provided"}), 400
            img_bytes = base64.b64decode(img_b64)
            age = float(data.get('age', 0))
            gender = data.get('gender', 'M').strip().upper()

        # encode gender
        gender_code = 1 if gender == 'F' else 0

        # preprocessing
        img_batch  = prep_image_for_flask(img_bytes)
        meta_batch = np.array([[age, gender_code]], dtype=np.float32)

        # predict
        pred_hgb = float(model.predict([img_batch, meta_batch], verbose=0)[0,0])
        threshold = 12.0
        status = "Anemia" if pred_hgb < threshold else "Normal"

        return jsonify({
            "predicted_hgb": pred_hgb,
            "status": status
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run with debug=False and use_reloader=False to avoid multiple processes in notebooks
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
