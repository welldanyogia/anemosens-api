"""
Google Colab Model Compatibility Tester
========================================
Script untuk test load AneMoSense models di Google Colab
dan identify versi TensorFlow/Keras yang dibutuhkan.

CARA PAKAI:
1. Upload script ini ke Google Colab
2. Upload model files (anemia_model_v1.h5, model_anemia_v2.h5)
3. Run semua cells
"""

# ============================================================
# CELL 1: Install Dependencies & Upload Models
# ============================================================

print("=" * 60)
print("AneMoSense Model Compatibility Tester")
print("=" * 60)

# Install h5py untuk inspect model
!pip install -q h5py

print("\n📤 Upload your model files:")
print("   - anemia_model_v1.h5")
print("   - model_anemia_v2.h5")
print("\nKlik 'Choose Files' di bawah untuk upload...")

from google.colab import files
uploaded = files.upload()

print(f"\n✅ Uploaded {len(uploaded)} file(s)")
for filename in uploaded.keys():
    print(f"   - {filename}")


# ============================================================
# CELL 2: Extract Model Metadata
# ============================================================

import h5py
import json

def inspect_model_h5(model_path):
    """Extract metadata from H5 model file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_path}")
    print(f"{'='*60}")

    try:
        with h5py.File(model_path, 'r') as f:
            print(f"\n📦 HDF5 Groups:")
            for key in f.keys():
                print(f"   - {key}")

            # Check Keras version
            if 'keras_version' in f.attrs:
                keras_version = f.attrs['keras_version']
                if isinstance(keras_version, bytes):
                    keras_version = keras_version.decode('utf-8')
                print(f"\n🔖 Keras Version: {keras_version}")

            # Check backend
            if 'backend' in f.attrs:
                backend = f.attrs['backend']
                if isinstance(backend, bytes):
                    backend = backend.decode('utf-8')
                print(f"🔖 Backend: {backend}")

            # Extract model config
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')

                config = json.loads(model_config)
                print(f"\n📋 Model Info:")
                print(f"   - Class: {config.get('class_name', 'Unknown')}")

                # Count layers
                if 'config' in config and 'layers' in config['config']:
                    layers = config['config']['layers']
                    print(f"   - Layers: {len(layers)}")

                    # Check for problematic configs
                    print(f"\n🔍 Layer Analysis:")
                    has_batch_shape = False
                    has_dtype_policy = False

                    for layer in layers:
                        layer_config = layer.get('config', {})

                        # Check InputLayer
                        if layer.get('class_name') == 'InputLayer':
                            if 'batch_shape' in layer_config:
                                has_batch_shape = True
                                print(f"   ⚠️  Found 'batch_shape' in {layer.get('name', 'unknown')}")
                            if 'batch_input_shape' in layer_config:
                                print(f"   ✅ Found 'batch_input_shape' in {layer.get('name', 'unknown')}")

                        # Check dtype
                        if 'dtype' in layer_config:
                            dtype_val = layer_config['dtype']
                            if isinstance(dtype_val, dict):
                                has_dtype_policy = True
                                print(f"   ⚠️  Found dict 'dtype' (DTypePolicy) in {layer.get('name', 'unknown')}")

                    # Recommendations
                    print(f"\n💡 Compatibility Issues:")
                    if has_batch_shape:
                        print(f"   ❌ Model uses 'batch_shape' (deprecated in TF 2.13+)")
                        print(f"      → Needs TensorFlow 2.10-2.12")
                    if has_dtype_policy:
                        print(f"   ❌ Model uses DTypePolicy dict format")
                        print(f"      → Needs TensorFlow 2.10-2.12")

                    if not has_batch_shape and not has_dtype_policy:
                        print(f"   ✅ Model config looks compatible with modern TF")

    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

# Inspect both models
for model_file in ['anemia_model_v1.h5', 'model_anemia_v2.h5']:
    try:
        inspect_model_h5(model_file)
    except FileNotFoundError:
        print(f"⚠️  File not found: {model_file}")


# ============================================================
# CELL 3: Test Load with Current TensorFlow Version
# ============================================================

import tensorflow as tf
print(f"\n{'='*60}")
print(f"Testing with Current TensorFlow Version")
print(f"{'='*60}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

def test_load_model(model_path):
    """Try to load model with current TF version."""
    print(f"\n🔄 Loading: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ SUCCESS! Model loaded successfully")

        # Print model summary
        print(f"\n📋 Model Summary:")
        model.summary()

        return model

    except TypeError as e:
        print(f"❌ TypeError: {e}")
        if 'batch_shape' in str(e):
            print(f"   → Issue: Model uses deprecated 'batch_shape' parameter")
            print(f"   → Solution: Downgrade to TensorFlow 2.10-2.12")
        return None

    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        if 'as_list' in str(e):
            print(f"   → Issue: Shape handling incompatibility")
            print(f"   → Solution: Try TensorFlow 2.10 or 2.11")
        return None

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return None

# Test both models
models = {}
for model_file in ['anemia_model_v1.h5', 'model_anemia_v2.h5']:
    try:
        model = test_load_model(model_file)
        if model:
            models[model_file] = model
    except FileNotFoundError:
        print(f"⚠️  File not found: {model_file}")


# ============================================================
# CELL 4: Try Different TensorFlow Versions
# ============================================================

print(f"\n{'='*60}")
print(f"Testing with Different TensorFlow Versions")
print(f"{'='*60}")

# List of versions to try (most likely to work based on Keras v19 indication)
versions_to_try = [
    "2.10.0",  # Keras 2.10 (likely match)
    "2.11.0",  # Keras 2.11
    "2.12.0",  # Keras 2.12 (last before big changes)
    "2.9.0",   # Older stable version
]

print("\n📋 Will test these TensorFlow versions:")
for v in versions_to_try:
    print(f"   - {v}")

print("\n⚠️  This will reinstall TensorFlow multiple times.")
print("    Each test takes ~1-2 minutes.")

import time

def test_tf_version(version, model_path):
    """Test loading model with specific TF version."""
    print(f"\n{'='*60}")
    print(f"Testing TensorFlow {version}")
    print(f"{'='*60}")

    # Uninstall current TF
    !pip uninstall -y -q tensorflow tensorflow-estimator keras

    # Install specific version
    print(f"📥 Installing TensorFlow {version}...")
    !pip install -q tensorflow=={version}

    # Reload TensorFlow
    import importlib
    import sys
    if 'tensorflow' in sys.modules:
        del sys.modules['tensorflow']

    import tensorflow as tf
    print(f"✅ Installed: TensorFlow {tf.__version__}, Keras {tf.keras.__version__}")

    # Try to load
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"\n🎉 SUCCESS! Model loaded with TensorFlow {version}")

        # Quick test prediction
        import numpy as np
        print(f"\n🧪 Testing prediction...")
        test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_meta = np.array([[0, 25]], dtype=np.float32)

        pred = model.predict([test_img, test_meta], verbose=0)
        print(f"✅ Prediction works! Output: {pred[0][0]:.2f}")

        return True

    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False

# Test V1 model with each version
print("\n" + "="*60)
print("TESTING MODEL V1 (anemia_model_v1.h5)")
print("="*60)

model_v1_path = 'anemia_model_v1.h5'
success_version = None

for version in versions_to_try:
    if test_tf_version(version, model_v1_path):
        success_version = version
        print(f"\n🎯 FOUND WORKING VERSION: TensorFlow {version}")
        break
    time.sleep(2)

if not success_version:
    print(f"\n❌ None of the tested versions worked.")
    print(f"   → Model may need to be re-saved from original training environment")


# ============================================================
# CELL 5: Generate Re-save Script
# ============================================================

if success_version:
    print(f"\n{'='*60}")
    print(f"Solution: Re-save Models Script")
    print(f"{'='*60}")

    script = f'''
# ========================================
# Re-save Models with TensorFlow {success_version}
# ========================================

# Install working version
!pip install tensorflow=={success_version} h5py

import tensorflow as tf
print(f"Using TensorFlow {{tf.__version__}}")

# Load and re-save Model V1
print("\\n🔄 Re-saving Model V1...")
model_v1 = tf.keras.models.load_model('anemia_model_v1.h5', compile=False)
model_v1.save('anemia_model_v1_fixed.h5', save_format='h5')
model_v1.save('anemia_model_v1_savedmodel', save_format='tf')
print("✅ Model V1 saved!")

# Load and re-save Model V2
print("\\n🔄 Re-saving Model V2...")
model_v2 = tf.keras.models.load_model('model_anemia_v2.h5', compile=False)
model_v2.save('model_anemia_v2_fixed.h5', save_format='h5')
model_v2.save('model_anemia_v2_savedmodel', save_format='tf')
print("✅ Model V2 saved!")

# Download fixed models
print("\\n📥 Downloading fixed models...")
from google.colab import files
files.download('anemia_model_v1_fixed.h5')
files.download('model_anemia_v2_fixed.h5')

print("\\n✅ DONE! Replace your old models with these fixed versions.")
'''

    print("\n📝 Copy and run this script in a NEW Colab notebook:")
    print("="*60)
    print(script)
    print("="*60)


# ============================================================
# CELL 6: Summary & Recommendations
# ============================================================

print(f"\n{'='*60}")
print(f"SUMMARY & RECOMMENDATIONS")
print(f"{'='*60}")

if success_version:
    print(f"\n✅ WORKING VERSION FOUND: TensorFlow {success_version}")
    print(f"\n📋 Next Steps:")
    print(f"   1. Use the re-save script above (CELL 5)")
    print(f"   2. Run it in a NEW Colab notebook")
    print(f"   3. Download the fixed model files")
    print(f"   4. Replace old models in your project")
    print(f"   5. Update Dockerfile to use TensorFlow 2.12.0 or later")
else:
    print(f"\n❌ NO WORKING VERSION FOUND")
    print(f"\n📋 Troubleshooting Options:")
    print(f"   1. Try loading in the ORIGINAL training environment")
    print(f"   2. Check if you have the original training script")
    print(f"   3. Try TensorFlow 2.8.0 or 2.7.0 (older versions)")
    print(f"   4. Last resort: Re-train the models from scratch")

print(f"\n{'='*60}")
print(f"For more help, check: MODEL_RESAVE_GUIDE.md")
print(f"{'='*60}")
