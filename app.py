#!/usr/bin/env python3
"""
app.py — Flask service for anemia prediction.

Model:
- Model V2 (model_anemia_v2.h5): Single accurate model with internal normalization.

Inputs:
- Input 1: Image (224, 224, 3)
- Input 2: Metadata [gender_code, age]
"""

# ─── Debugpy / Colab repr hack ──────────────────────────────────────────────────
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
    pass

# ─── Standard imports ──────────────────────────────────────────────────────────
import logging
import os
import base64
import re
from datetime import datetime, timezone
import json
import h5py
import tempfile
import shutil

import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

logging.getLogger().setLevel(logging.ERROR)

# Model paths
MODEL_PATH = 'model_anemia_v2.keras'

# Input image size
IMG_SIZE = (224, 224)

# Anemia threshold (Hgb in g/dL)
ANEMIA_THRESHOLD = 12.0

# ─── Model Loading with Compatibility Layer ────────────────────────────────────

def load_model_with_compat(model_path: str):
    """
    Load Keras model with compatibility for older model formats.
    Handles the batch_shape -> shape migration in InputLayer.
    """
    try:
        # First, try standard loading
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as e:
        if 'batch_shape' not in str(e):
            raise  # Re-raise if it's a different error

        print(f"   ⚠️  Detected legacy model format, applying compatibility fix...")

        # Read the model config
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError("Model config not found in h5 file")

            # Parse the config
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            config = json.loads(model_config)

        # Fix layer configs (convert incompatible formats)
        def fix_layer_config(layer_cfg):
            cfg = layer_cfg.get('config', {})

            # Fix InputLayer batch_shape issues
            if layer_cfg.get('class_name') == 'InputLayer':
                # Handle batch_shape (old format)
                if 'batch_shape' in cfg:
                    batch_shape = cfg.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        cfg['batch_input_shape'] = batch_shape

                # Remove 'shape' if present (not supported in some versions)
                if 'shape' in cfg:
                    shape = cfg.pop('shape')
                    if shape:
                        cfg['batch_input_shape'] = [None] + list(shape)

                # Ensure we have batch_input_shape
                if 'batch_input_shape' not in cfg and 'input_shape' in cfg:
                    cfg['batch_input_shape'] = [None] + list(cfg.pop('input_shape'))

            # Fix DTypePolicy issues across all layers
            if 'dtype' in cfg:
                dtype_val = cfg['dtype']
                # If dtype is a dict with DTypePolicy structure, extract the actual dtype
                if isinstance(dtype_val, dict):
                    if 'config' in dtype_val and 'name' in dtype_val['config']:
                        # Extract just the dtype name (e.g., 'float32')
                        cfg['dtype'] = dtype_val['config']['name']
                    elif 'class_name' in dtype_val:
                        # Fallback: use class_name if no config.name
                        cfg['dtype'] = 'float32'  # Default to float32

            # Fix initializers if they have module structure
            for key in ['kernel_initializer', 'bias_initializer', 'gamma_initializer',
                       'beta_initializer', 'moving_mean_initializer', 'moving_variance_initializer']:
                if key in cfg and isinstance(cfg[key], dict):
                    if 'module' in cfg[key]:
                        # Simplify to just class_name and config
                        cfg[key] = {
                            'class_name': cfg[key].get('class_name', 'GlorotUniform'),
                            'config': cfg[key].get('config', {})
                        }

            return layer_cfg

        # Process all layers
        if 'config' in config:
            if 'layers' in config['config']:
                for layer in config['config']['layers']:
                    fix_layer_config(layer)
            elif isinstance(config['config'], dict):
                fix_layer_config(config['config'])

        # Create a temporary patched model file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Copy original file to temp
            shutil.copy2(model_path, tmp_path)

            # Update the config in the temp file
            with h5py.File(tmp_path, 'r+') as f:
                f.attrs.modify('model_config', json.dumps(config).encode('utf-8'))

            # Load from the patched temp file
            model = tf.keras.models.load_model(tmp_path, compile=False)
            print(f"   ✅ Legacy model loaded successfully")
            return model
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


print("🔄 Loading Model V2 (Accurate)...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"   ✅ Model loaded: {MODEL_PATH}")


def validate_age(age_value) -> tuple:
    """Validate age input is a valid number between 0 and 120."""
    try:
        age = float(age_value)
        if age < 0 or age > 120:
            return None, "Invalid age"
        return age, None
    except (ValueError, TypeError):
        return None, "Invalid age"


def validate_file_size(file_obj, max_size_mb=5) -> bool:
    """Check if file size is within the allowed limit."""
    file_obj.seek(0, os.SEEK_END)
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    max_bytes = max_size_mb * 1024 * 1024
    return size_bytes <= max_bytes


def decode_image(img_data: bytes) -> np.ndarray:
    """
    Decode raw JPEG/PNG bytes to RGB numpy array.
    Returns: np.ndarray of shape (H, W, 3) in RGB format, dtype uint8
    """
    arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    img = cv2.resize(img, IMG_SIZE)
    return img


def preprocess_for_model(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for Model V2 (EfficientNet-style).
    """
    img_float = img.astype(np.float32)
    img_processed = efficientnet_preprocess(img_float)
    return np.expand_dims(img_processed, axis=0)


def get_status(hgb: float, gender_code: int) -> dict:
    """
    Determine anemia status based on Hgb and gender.
    
    WHO thresholds:
    - Male: < 13.0 g/dL = Anemia
    - Female: < 12.0 g/dL = Anemia
    
    Returns status info dict.
    """
    threshold = 13.0 if gender_code == 0 else 12.0  # Male vs Female
    
    if hgb < threshold - 2:
        severity = "moderate"
    elif hgb < threshold:
        severity = "mild"
    else:
        severity = "normal"
    
    is_anemia = hgb < threshold
    
    return {
        "is_anemia": is_anemia,
        "status": "Anemia" if is_anemia else "Normal",
        "severity": severity,
        "threshold": threshold
    }


def predict_single_model(model_instance, img_batch: np.ndarray, meta_batch: np.ndarray) -> float:
    """Run prediction on the model and return Hgb value."""
    prediction = model_instance.predict([img_batch, meta_batch], verbose=0)
    return float(prediction[0, 0])


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Accepts:
    - multipart/form-data with 'image', 'gender', 'age'
    - application/json with base64 'image', 'gender', 'age'
    
    Returns:
    - meta: timestamp and request info
    - predicted_hgb: Hgb value
    - status: Anemia status
    """
    try:
        # ─── Parse Input ───────────────────────────────────────────────────
        if request.content_type and request.content_type.startswith('multipart/'):
            img_file = request.files.get('image') or request.files.get('file')
            if not img_file:
                return jsonify({"error": "No image file provided"}), 400

            if not img_file.filename or img_file.filename.strip() == '':
                return jsonify({"error": "Invalid filename"}), 400

            if not img_file.mimetype or not img_file.mimetype.startswith('image/'):
                return jsonify({"error": "Invalid file type"}), 400

            if not validate_file_size(img_file, max_size_mb=5):
                return jsonify({"error": "File too large"}), 413

            img_bytes = img_file.read()

            age_str = request.form.get('age', '0')
            age, error = validate_age(age_str)
            if error:
                return jsonify({"error": error}), 400

            gender = request.form.get('gender', 'M').strip().upper()

        else:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON or multipart/form-data"}), 415
            
            data = request.get_json()
            img_b64 = data.get('image')
            if not img_b64:
                return jsonify({"error": "No image data provided"}), 400

            img_bytes = base64.b64decode(img_b64)

            max_bytes = 5 * 1024 * 1024
            if len(img_bytes) > max_bytes:
                return jsonify({"error": "File too large"}), 413

            age_str = data.get('age', '0')
            age, error = validate_age(age_str)
            if error:
                return jsonify({"error": error}), 400

            gender = data.get('gender', 'M').strip().upper()

        # ─── Encode Gender ─────────────────────────────────────────────────
        # Handle both "M"/"F" and "0"/"1" formats
        if gender in ['F', '1', '1.0']:
            gender_code = 1  # Female
        else:
            gender_code = 0  # Male

        # ─── Decode & Preprocess Image ─────────────────────────────────────
        img_rgb = decode_image(img_bytes)
        
        # Preprocess for model
        img_batch = preprocess_for_model(img_rgb)
        
        # Metadata batch
        # Format: [gender_code, age] based on training
        meta_batch = np.array([[gender_code, age]], dtype=np.float32)

        # ─── Run Prediction ────────────────────────────────────────────────
        try:
            hgb = predict_single_model(model, img_batch, meta_batch)
            hgb = round(hgb, 2)
            status_info = get_status(hgb, gender_code)
        except Exception as e:
            print(f"Model Error: {e}")
            return jsonify({
                "error": "Prediction failed",
                "details": str(e)
            }), 500

        # ─── Build Response ────────────────────────────────────────────────
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return jsonify({
            # Metadata
            "meta": {
                "timestamp": timestamp,
                "input": {
                    "gender": "Female" if gender_code == 1 else "Male",
                    "gender_code": gender_code,
                    "age": age
                },
                "model": "V2 (Single)"
            },
            
            # Result
            "predicted_hgb": hgb,
            "status": status_info["status"],
            "severity": status_info["severity"],
            "is_anemia": status_info["is_anemia"],
            
            # Legacy campatibility (optional, but good to keep structure for safety)
            "predictions": {
                "v2": {
                     "hgb": hgb,
                     "status": status_info["status"],
                     "severity": status_info["severity"],
                     "is_anemia": status_info["is_anemia"]
                }
            }
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"INTERNAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Terjadi kesalahan sistem saat memproses gambar"}), 500


# ─── Health & Info Endpoints ───────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """Health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH
    }), 200


@app.route('/models', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    return jsonify({
        "models": [
            {
                "name": "Model V2",
                "path": MODEL_PATH,
                "description": "Accurate anemia prediction model.",
                "input_shape": {
                    "image": [224, 224, 3],
                    "metadata": ["gender_code", "age"]
                },
                "preprocessing": "EfficientNet-style"
            }
        ],
        "endpoints": {
            "/predict": "Prediction endpoint"
        }
    }), 200


# ─── API Documentation ─────────────────────────────────────────────────────────

@app.route('/docs', methods=['GET'])
def swagger_ui():
    """Serve Swagger UI for API documentation."""
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
    """Serve OpenAPI specification file."""
    try:
        with open('openapi.yaml', 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/yaml'}
    except FileNotFoundError:
        return jsonify({"error": "OpenAPI spec not found"}), 404


# ─── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
