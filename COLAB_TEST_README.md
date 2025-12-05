# 🧪 Google Colab Model Testing Guide

Panduan untuk test load AneMoSense models di Google Colab dan identify versi TensorFlow yang kompatibel.

## 🚀 Quick Start

### Step 1: Buka Google Colab
1. Buka https://colab.research.google.com
2. Klik **File → New Notebook**

### Step 2: Upload Script
Ada 2 cara:

#### Option A: Upload File Script
1. Upload `test_model_colab.py` ke Colab
2. Run dengan: `!python test_model_colab.py`

#### Option B: Copy-Paste Langsung (RECOMMENDED)
Copy-paste code dari `test_model_colab.py` ke cells di Colab.

---

## 📋 Step-by-Step Execution

### CELL 1: Upload Model Files
```python
# Install h5py
!pip install -q h5py

# Upload models
from google.colab import files
print("📤 Upload your model files:")
uploaded = files.upload()

print(f"\n✅ Uploaded {len(uploaded)} file(s)")
for filename in uploaded.keys():
    print(f"   - {filename}")
```

**Action**: Klik tombol upload dan pilih:
- `anemia_model_v1.h5`
- `model_anemia_v2.h5`

---

### CELL 2: Inspect Model Metadata
```python
import h5py
import json

def inspect_model_h5(model_path):
    """Extract metadata from H5 model file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {model_path}")
    print(f"{'='*60}")

    with h5py.File(model_path, 'r') as f:
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
            print(f"\n📋 Model Class: {config.get('class_name', 'Unknown')}")

            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
                print(f"📋 Total Layers: {len(layers)}")

# Inspect both models
for model_file in ['anemia_model_v1.h5', 'model_anemia_v2.h5']:
    inspect_model_h5(model_file)
```

**Expected Output**: Versi Keras yang digunakan (e.g., "2.10.0" atau "2.19")

---

### CELL 3: Test with Current TensorFlow
```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")

def test_load(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ {model_path} loaded successfully!")
        model.summary()
        return True
    except Exception as e:
        print(f"❌ {model_path} failed: {type(e).__name__}: {e}")
        return False

# Test both models
test_load('anemia_model_v1.h5')
test_load('model_anemia_v2.h5')
```

**Possible Results**:
- ✅ Success → Model kompatibel dengan TF di Colab (saat ini 2.15+)
- ❌ Error → Lanjut ke CELL 4

---

### CELL 4: Test TensorFlow 2.10 (Most Likely to Work)
```python
# Uninstall current TF
!pip uninstall -y -q tensorflow tensorflow-estimator keras

# Install TF 2.10
print("📥 Installing TensorFlow 2.10...")
!pip install -q tensorflow==2.10.0

# Reload TensorFlow
import importlib
import sys
if 'tensorflow' in sys.modules:
    del sys.modules['tensorflow']

import tensorflow as tf
print(f"✅ TensorFlow {tf.__version__}, Keras {tf.keras.__version__}")

# Try to load
model_v1 = tf.keras.models.load_model('anemia_model_v1.h5', compile=False)
print("✅ Model V1 loaded!")

model_v2 = tf.keras.models.load_model('model_anemia_v2.h5', compile=False)
print("✅ Model V2 loaded!")

# Test prediction
import numpy as np
test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
test_meta = np.array([[0, 25]], dtype=np.float32)

pred = model_v1.predict([test_img, test_meta], verbose=0)
print(f"✅ Prediction works! HGB: {pred[0][0]:.2f}")
```

**If this works**: TensorFlow 2.10 is your target version!

---

### CELL 5: Re-save Models (JIKA CELL 4 BERHASIL)
```python
print("🔄 Re-saving models in compatible format...")

# Save Model V1
model_v1.save('anemia_model_v1_fixed.h5', save_format='h5')
model_v1.save('anemia_model_v1_savedmodel', save_format='tf')
print("✅ Model V1 saved!")

# Save Model V2
model_v2.save('model_anemia_v2_fixed.h5', save_format='h5')
model_v2.save('model_anemia_v2_savedmodel', save_format='tf')
print("✅ Model V2 saved!")

# Zip SavedModel directories
!zip -r anemia_model_v1_savedmodel.zip anemia_model_v1_savedmodel
!zip -r model_anemia_v2_savedmodel.zip model_anemia_v2_savedmodel

# Download fixed models
from google.colab import files

print("\n📥 Downloading fixed models...")
files.download('anemia_model_v1_fixed.h5')
files.download('model_anemia_v2_fixed.h5')
files.download('anemia_model_v1_savedmodel.zip')
files.download('model_anemia_v2_savedmodel.zip')

print("\n✅ DONE! Extract ZIP files dan replace models di project Anda.")
```

---

## 🎯 Expected Workflow

1. **CELL 1** → Upload models ✅
2. **CELL 2** → See Keras version (probably "2.10.0" or "2.11.0") 📋
3. **CELL 3** → Fail dengan TF 2.15 (default di Colab) ❌
4. **CELL 4** → Success dengan TF 2.10 ✅
5. **CELL 5** → Download fixed models 📥

---

## 📦 What You'll Get

After running CELL 5, you'll download:

```
anemia_model_v1_fixed.h5              # 11 MB - untuk app.py
model_anemia_v2_fixed.h5              # 49 MB - untuk app.py
anemia_model_v1_savedmodel.zip        # Alternative format
model_anemia_v2_savedmodel.zip        # Alternative format
```

---

## 🔧 Update Project Files

### 1. Replace Model Files
```bash
# Di local project
mv anemia_model_v1_fixed.h5 anemia_model_v1.h5
mv model_anemia_v2_fixed.h5 model_anemia_v2.h5
```

### 2. Update `app.py` (No Change Needed)
```python
# Model paths tetap sama
MODEL_V1_PATH = 'anemia_model_v1.h5'
MODEL_V2_PATH = 'model_anemia_v2.h5'
```

### 3. Update `requirements.txt`
```
tensorflow==2.12.0  # Atau gunakan 2.13.0 (lebih stabil)
```

### 4. Rebuild Docker
```bash
docker build -f Dockerfile.dev -t anemosense-dev .
docker run -p 5000:5000 anemosense-dev
```

---

## 🐛 Troubleshooting

### Error: "File not found"
**Solution**: Pastikan model files sudah di-upload di CELL 1

### Error: "Module 'tensorflow' has no attribute 'keras'"
**Solution**: Restart runtime → Runtime → Restart runtime

### Download tidak muncul
**Solution**:
```python
# Check files
!ls -lh *.h5 *.zip
```

### Model tetap error setelah re-save
**Solution**: Gunakan SavedModel format instead of H5:
```python
# Di app.py
MODEL_V1_PATH = 'anemia_model_v1_savedmodel'
MODEL_V2_PATH = 'model_anemia_v2_savedmodel'
```

---

## ✅ Success Checklist

- [ ] Models uploaded to Colab
- [ ] Keras version identified (CELL 2)
- [ ] Models load successfully with TF 2.10 (CELL 4)
- [ ] Re-save completed (CELL 5)
- [ ] Fixed models downloaded
- [ ] Models replaced in local project
- [ ] Docker build successful
- [ ] Flask app starts without errors
- [ ] `/health` endpoint returns OK
- [ ] `/predict` endpoint works

---

## 📞 Alternative: Try Other Versions

Jika TF 2.10 tidak work, coba versions lain di CELL 4:

```python
# Try TF 2.11
!pip install -q tensorflow==2.11.0

# Try TF 2.9
!pip install -q tensorflow==2.9.0

# Try TF 2.8
!pip install -q tensorflow==2.8.0
```

---

**Good luck! 🚀**
