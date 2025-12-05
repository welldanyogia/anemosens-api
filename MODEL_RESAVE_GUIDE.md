# 📘 Model Re-Save Guide

Guide lengkap untuk re-save models AneMoSense agar kompatibel dengan TensorFlow modern.

## 🎯 Tujuan

Script `resave_models.py` akan:
- ✅ Load models lama dengan berbagai strategi
- ✅ Convert ke format yang kompatibel
- ✅ Save dalam 3 format berbeda (H5, Keras, SavedModel)
- ✅ Test hasil konversi

## 📋 Prerequisites

### Option 1: Gunakan Environment Training Asli (RECOMMENDED)

Jika Anda masih punya environment yang digunakan untuk training model:

```bash
# Aktifkan environment training
conda activate your_training_env  # atau source venv/bin/activate

# Jalankan script
python resave_models.py
```

### Option 2: Setup Environment Baru dengan TensorFlow Compatible

```bash
# Buat virtual environment baru
python -m venv venv_resave
source venv_resave/bin/activate  # Windows: venv_resave\Scripts\activate

# Install TensorFlow versi yang kompatibel dengan models
# Coba beberapa versi ini sampai berhasil load:
pip install tensorflow==2.10.0  # Atau 2.11.0, 2.12.0

# Install h5py
pip install h5py
```

## 🚀 Cara Penggunaan

### Basic Usage

```bash
# Default: akan process anemia_model_v1.h5 dan model_anemia_v2.h5
python resave_models.py

# Output akan tersimpan di folder: models_fixed/
```

### Advanced Usage

```bash
# Custom model paths
python resave_models.py \
  --v1 path/to/your/model_v1.h5 \
  --v2 path/to/your/model_v2.h5

# Custom output directory
python resave_models.py --output-dir my_fixed_models

# Dengan testing setelah save
python resave_models.py --test

# Kombinasi semua options
python resave_models.py \
  --v1 anemia_model_v1.h5 \
  --v2 model_anemia_v2.h5 \
  --output-dir models_fixed \
  --test
```

## 📂 Output Files

Script akan generate 3 format untuk setiap model:

```
models_fixed/
├── anemia_model_v1_fixed.h5              # HDF5 format (paling kompatibel)
├── anemia_model_v1_fixed.keras           # Keras native format
├── anemia_model_v1_savedmodel/           # TensorFlow SavedModel (paling reliable)
│   ├── saved_model.pb
│   ├── variables/
│   └── assets/
├── model_anemia_v2_fixed.h5
├── model_anemia_v2_fixed.keras
└── model_anemia_v2_savedmodel/
    ├── saved_model.pb
    ├── variables/
    └── assets/
```

## 🔧 Update app.py

Setelah re-save berhasil, update `app.py`:

### Option A: Gunakan HDF5 Format (Recommended)

```python
# Di app.py, line 52-53
MODEL_V1_PATH = 'models_fixed/anemia_model_v1_fixed.h5'
MODEL_V2_PATH = 'models_fixed/model_anemia_v2_fixed.h5'
```

### Option B: Gunakan SavedModel Format (Most Reliable)

```python
# Di app.py, line 52-53
MODEL_V1_PATH = 'models_fixed/anemia_model_v1_savedmodel'
MODEL_V2_PATH = 'models_fixed/model_anemia_v2_savedmodel'
```

## 🧪 Testing

### Test 1: Run Script dengan --test Flag

```bash
python resave_models.py --test
```

### Test 2: Test Load Manual

```python
import tensorflow as tf

# Test load H5
model_v1 = tf.keras.models.load_model('models_fixed/anemia_model_v1_fixed.h5', compile=False)
print("✅ V1 H5 loaded!")

# Test load SavedModel
model_v2 = tf.keras.models.load_model('models_fixed/model_anemia_v2_savedmodel', compile=False)
print("✅ V2 SavedModel loaded!")

# Test prediction
import numpy as np
test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
test_meta = np.array([[0, 25]], dtype=np.float32)

pred = model_v1.predict([test_img, test_meta])
print(f"✅ Prediction works: {pred[0][0]}")
```

### Test 3: Run Flask App

```bash
# Test locally
python app.py

# Test endpoint
curl -X GET http://localhost:5000/health
```

## 🐛 Troubleshooting

### Error: "Cannot load model"

**Solution:**
Try different TensorFlow versions:

```bash
# Coba TF 2.10
pip install tensorflow==2.10.0
python resave_models.py

# Jika gagal, coba TF 2.11
pip install tensorflow==2.11.0
python resave_models.py

# Atau TF 2.12
pip install tensorflow==2.12.0
python resave_models.py
```

### Error: "All loading strategies failed"

**Penyebab:** Model sangat custom atau corrupt

**Solution:**
1. Check apakah file model corrupt:
   ```bash
   python -c "import h5py; f = h5py.File('anemia_model_v1.h5', 'r'); print(list(f.keys()))"
   ```

2. Coba load di environment training asli

3. Re-train model (last resort)

### Error: "Out of memory"

**Solution:**
```bash
# Set memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Atau gunakan CPU only
export CUDA_VISIBLE_DEVICES=-1
python resave_models.py
```

## 📦 Update Dockerfile

Setelah re-save, update Dockerfile untuk copy fixed models:

```dockerfile
# Dockerfile
COPY models_fixed/anemia_model_v1_fixed.h5 ./anemia_model_v1.h5
COPY models_fixed/model_anemia_v2_fixed.h5 ./model_anemia_v2.h5

# Atau jika pakai SavedModel:
COPY models_fixed/anemia_model_v1_savedmodel ./anemia_model_v1_savedmodel
COPY models_fixed/model_anemia_v2_savedmodel ./model_anemia_v2_savedmodel
```

## ✅ Verification Checklist

- [ ] Script runs without errors
- [ ] Models loaded successfully
- [ ] Output files created in `models_fixed/`
- [ ] Test load works (with `--test` flag)
- [ ] `app.py` updated dengan path baru
- [ ] Flask app starts successfully
- [ ] `/health` endpoint returns OK
- [ ] `/predict` endpoint works dengan test image
- [ ] Docker build succeeds
- [ ] Docker container runs successfully

## 🎯 Expected Output

```
============================================================
  AneMoSense Model Re-Saver
  Converting legacy models to compatible format
============================================================
✅ TensorFlow version: 2.12.0
✅ Keras version: 2.12.0

============================================================
PROCESSING MODEL V1 (Lightweight)
============================================================
============================================================
Loading model: anemia_model_v1.h5
============================================================
📥 Strategy 1: Direct load...
✅ Model loaded successfully with direct load!

============================================================
Saving model: anemia_model_v1
============================================================
💾 Saving as HDF5: models_fixed/anemia_model_v1_fixed.h5
✅ HDF5 saved successfully!
   Size: 11.12 MB
💾 Saving as SavedModel: models_fixed/anemia_model_v1_savedmodel
✅ SavedModel saved successfully!
   Size: 11.50 MB

... (similar for Model V2)

============================================================
SUMMARY
============================================================
✅ Output directory: /path/to/models_fixed
✅ Files created: 6
   - anemia_model_v1_fixed.h5 (11.12 MB)
   - anemia_model_v1_savedmodel/ (11.50 MB)
   - model_anemia_v2_fixed.h5 (48.95 MB)
   - model_anemia_v2_savedmodel/ (49.20 MB)
```

## 📞 Support

Jika masih ada masalah:
1. Check log output dari script
2. Verify TensorFlow version compatibility
3. Pastikan file model asli tidak corrupt
4. Coba di environment training asli

---

**Good luck! 🚀**
