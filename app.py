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
import re

import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────────────────────────────────
# Load environment variables from .env file
load_dotenv()

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

# ─── CORS Configuration ─────────────────────────────────────────────────────────
# Allow requests from localhost (development) and production domain
# Regex pattern allows localhost with any port number (e.g., :3000, :5000, :8080)
CORS(app,
     origins=[
         re.compile(r"^http://localhost(:\d+)?$"),  # Match localhost with optional port
         "https://anemosense.webranastore.com"
     ],
     supports_credentials=True)

def validate_age(age_value) -> tuple:
    """
    Validate age input is a valid number between 0 and 120.
    Returns (age_float, error_message).
    """
    try:
        age = float(age_value)
        if age < 0 or age > 120:
            return None, "Invalid age"
        return age, None
    except (ValueError, TypeError):
        return None, "Invalid age"


def validate_file_size(file_obj, max_size_mb=5) -> bool:
    """
    Check if file size is within the allowed limit.
    Uses seek/tell to check size without loading entire file into memory.
    """
    file_obj.seek(0, os.SEEK_END)
    size_bytes = file_obj.tell()
    file_obj.seek(0)  # Reset to beginning

    max_bytes = max_size_mb * 1024 * 1024
    return size_bytes <= max_bytes


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

            # Validate filename
            if not img_file.filename or img_file.filename.strip() == '':
                return jsonify({"error": "Invalid filename"}), 400

            # Validate MIME type
            if not img_file.mimetype or not img_file.mimetype.startswith('image/'):
                return jsonify({"error": "Invalid file type"}), 400

            # Validate file size (max 5MB)
            if not validate_file_size(img_file, max_size_mb=5):
                return jsonify({"error": "File too large"}), 413

            img_bytes = img_file.read()

            # Validate age
            age_str = request.form.get('age', '0')
            age, error = validate_age(age_str)
            if error:
                return jsonify({"error": error}), 400

            gender = request.form.get('gender', 'M').strip().upper()
        # —————— CASE 2: application/json ——————
        else:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON or multipart/form-data"}), 415
            data = request.get_json()
            img_b64 = data.get('image')
            if not img_b64:
                return jsonify({"error": "No image data provided"}), 400

            # Decode base64 image
            img_bytes = base64.b64decode(img_b64)

            # Validate decoded image size (max 5MB)
            max_bytes = 5 * 1024 * 1024
            if len(img_bytes) > max_bytes:
                return jsonify({"error": "File too large"}), 413

            # Validate age
            age_str = data.get('age', '0')
            age, error = validate_age(age_str)
            if error:
                return jsonify({"error": error}), 400

            gender = data.get('gender', 'M').strip().upper()

        # Encode gender - handle both formats:
        # "M"/"F" (standard) or "0"/"1" (legacy mobile app)
        if gender in ['F', '1', '1.0']:
            gender_code = 1  # Female
        else:
            gender_code = 0  # Male (default for "M", "0", "0.0", or any other value)

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
        # Validation errors are safe to expose
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Log the full error for internal debugging
        print(f"INTERNAL ERROR: {e}")
        # Return generic message to prevent information disclosure
        return jsonify({"error": "Terjadi kesalahan sistem saat memproses gambar"}), 500


# ─── Health-check route ───────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health-check endpoint to verify the application is running.
    """
    return jsonify({"status": "ok"}), 200


# ─── API Documentation routes ─────────────────────────────────────────────────
@app.route('/docs', methods=['GET'])
def swagger_ui():
    """
    Serve Swagger UI for API documentation.
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AneMoSense API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
        <style>
            body { margin: 0; padding: 0; }
            .swagger-ui .topbar { display: none; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: "/openapi.yaml",
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    layout: "BaseLayout",
                    deepLinking: true,
                    showExtensions: true,
                    showCommonExtensions: true
                });
            };
        </script>
    </body>
    </html>
    ''', 200, {'Content-Type': 'text/html'}


@app.route('/openapi.yaml', methods=['GET'])
def openapi_spec():
    """
    Serve OpenAPI specification file.
    """
    try:
        with open('openapi.yaml', 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/yaml'}
    except FileNotFoundError:
        return jsonify({"error": "OpenAPI spec not found"}), 404

# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run with debug=False and use_reloader=False to avoid multiple processes in notebooks
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
