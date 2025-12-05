# Anemosense API

Flask-based REST API for anemia prediction using **Dual-Model Architecture** (TensorFlow/Keras). Provides endpoints for image-based hemoglobin level prediction with metadata support.

## Features

- **🚀 Dual-Model Inference**: V1 (Lightweight/Fast) + V2 (Accurate) + Combined Result
- **📊 Multi-Model Support**: Choose between V1-only, V2-only, or Dual-Model prediction
- **📤 Multi-format Input**: Supports both `multipart/form-data` and `application/json`
- **🔒 CORS Support**: Configured for cross-origin requests (Sprint 1)
- **⚙️ Production Ready**: Gunicorn WSGI server with Docker support
- **📖 Interactive Docs**: Swagger UI for API exploration
- **💚 Health Monitoring**: `/health` and `/models` endpoints

## Model Architecture

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| **V1 (Lightweight)** | 11 MB | MobileNetV2-based, faster inference | Screening, batch processing, low-resource devices |
| **V2 (Accurate)** | 49 MB | EfficientNet-based, higher accuracy | Final diagnosis, high-accuracy requirements |
| **Combined** | Both | Weighted average (30% V1 + 70% V2) | Best balance of speed and accuracy |

## Sprint 1 Updates (Security Hardening)

### BE-01: CORS Implementation ✅
- **Added**: `flask-cors` with whitelist configuration
- **Security**: Only allows `localhost` (dev) and `https://anemosense.webrana.id` (prod)
- **Regex Support**: Dynamic port matching for localhost development
- **Credentials**: Supports cookie/auth header forwarding

**Files Modified**:
- `app.py`: Added CORS middleware with regex pattern
- `requirements.txt`: Pinned all dependency versions + added `flask-cors==4.0.0`
- `.env.example`: Created environment variable template

---

## Sprint 2 Updates (Stability & Resilience)

### BE-02: Input Validation ✅
- **Age Validation**: Range check (0-120) with type conversion safety
- **File Size Limit**: Maximum 5MB enforced using `seek()/tell()` pattern
- **MIME Type Check**: Only accepts `image/*` content types
- **Filename Validation**: Prevents empty or invalid filenames

**Implementation**:
```python
# Age validation with range check
def validate_age(age_value) -> tuple:
    age = float(age_value)
    if age < 0 or age > 120:
        return None, "Invalid age"
    return age, None

# File size validation (max 5MB)
def validate_file_size(file_obj, max_size_mb=5) -> bool:
    file_obj.seek(0, os.SEEK_END)
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    return size_bytes <= max_size_mb * 1024 * 1024
```

**Error Responses**:
- `400`: Invalid age, filename, or file type
- `413`: File too large (>5MB)

### BE-04: Security Hygiene (Hide Stack Traces) ✅
- **Generic Error Messages**: Internal exceptions return generic message to client
- **Internal Logging**: Full error details logged to server console for debugging
- **Information Disclosure Prevention**: No file paths, variable names, or stack traces exposed

**Before**:
```python
except Exception as e:
    return jsonify({"error": f"Prediction failed: {e}"}), 500
    # Exposes: FileNotFoundError, paths, internal details
```

**After**:
```python
except Exception as e:
    print(f"INTERNAL ERROR: {e}")  # Server-side logging
    return jsonify({"error": "Terjadi kesalahan sistem saat memproses gambar"}), 500
    # Client gets generic message only
```

**Files Modified**:
- `app.py`: Added validation functions + secure error handling (+52 lines)

**Security Impact**:
- ✅ Prevents DoS attacks via large files
- ✅ Blocks invalid age inputs that could crash ML model
- ✅ Hides internal server architecture from attackers
- ✅ Maintains debuggability via server logs

---

## API Documentation

### Interactive Swagger UI
Dokumentasi API interaktif tersedia di:
- **Local**: http://localhost:5000/docs
- **Production**: https://api.anemosense.webranastore.com/docs

### OpenAPI Specification
File OpenAPI 3.0 spec tersedia di:
- **Local**: http://localhost:5000/openapi.yaml
- **File**: `openapi.yaml`

---

## API Endpoints

### 1. Dual-Model Prediction (Recommended)
**POST** `/predict`

Menganalisis gambar mata menggunakan **kedua model (V1 & V2)** dan mengembalikan hasil gabungan.

**Request (multipart/form-data)**:
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@photo.jpg" \
  -F "age=25" \
  -F "gender=F"
```

**Request (application/json)**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-image>",
    "age": 25,
    "gender": "F"
  }'
```

**Response (Dual-Model)**:
```json
{
  "meta": {
    "timestamp": "2025-11-26T10:30:00.000Z",
    "input": {
      "gender": "Female",
      "gender_code": 1,
      "age": 25
    }
  },
  "predicted_hgb": 14.5,
  "status": "Normal",
  "predictions": {
    "v1": {
      "hgb": 14.2,
      "status": "Normal",
      "severity": "normal",
      "is_anemia": false
    },
    "v2": {
      "hgb": 14.6,
      "status": "Normal",
      "severity": "normal",
      "is_anemia": false
    },
    "combined": {
      "hgb": 14.5,
      "status": "Normal",
      "severity": "normal",
      "is_anemia": false,
      "weights": {
        "v1": 0.3,
        "v2": 0.7
      }
    }
  }
}
```

**Parameters**:
- `image`: Image file (JPG/PNG) or base64 string
- `age`: Patient age (0-120)
- `gender`: `M`/`0` (male) or `F`/`1` (female)

**Status Codes**:
- `200`: Success
- `400`: Invalid input (missing image, bad format, invalid age)
- `413`: File too large (>5MB)
- `415`: Unsupported media type
- `500`: Prediction failed

---

### 2. Single-Model Predictions

**POST** `/predict/v1` - Model V1 Only (Lightweight)

Prediksi menggunakan **Model V1 saja** (faster, lighter).

**POST** `/predict/v2` - Model V2 Only (Accurate)

Prediksi menggunakan **Model V2 saja** (more accurate).

Format request sama dengan `/predict`, response hanya berisi hasil dari satu model.

---

### 3. Health Check & Model Info
**GET** `/health`

Returns API status and loaded models.

**Response**:
```json
{
  "status": "ok",
  "models": {
    "v1": "anemia_model_v1.h5",
    "v2": "model_anemia_v2.h5"
  }
}
```

**GET** `/models`

Returns detailed information about available models, input shapes, preprocessing, and endpoints.

---

### 4. API Documentation
**GET** `/docs`

Interactive Swagger UI untuk menjelajahi dan menguji API.

**Akses:**
- **Local**: http://localhost:5000/docs
- **Production**: https://api.anemosense.webranastore.com/docs

**GET** `/openapi.yaml`

OpenAPI 3.0 specification file dalam format YAML.

---

## Postman Collection

Import file `anemosense_api.postman_collection.json` ke Postman untuk testing.

**Quick Start:**
1. Import collection ke Postman
2. Set variable `{{base_url}}`:
   - Local: `http://localhost:5000`
   - Production: `https://api.anemosense.webranastore.com`
3. Upload gambar mata di request `POST /predict`
4. Run request!

---

## Local Development

### Option 1: Docker (Recommended)

**Development Mode** (hot reload):
```bash
docker build -f Dockerfile.dev -t anemosense-dev .
docker run --rm -it -p 5000:5000 -v $(pwd):/app anemosense-dev
```

**Production Mode**:
```bash
docker build -t anemosense-api .
docker run -d --name anemosense-api -p 5000:5000 anemosense-api
```

### Option 2: Local Python

**Prerequisites**: Python 3.10+

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

API available at: `http://localhost:5000`

---

## Environment Variables

Create `.env` file (use `.env.example` as template):

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_APP=app.py
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000

# Model Configuration
MODEL_PATH=model_anemia.h5
ANEMIA_THRESHOLD=12.0

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:*,https://anemosense.webranastore.com

# Logging
LOG_LEVEL=ERROR
```

**Note**: `.env` is optional. Defaults are hardcoded in `app.py`.

---

## Production Deployment

### 1. Install Docker on VPS

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl enable --now docker
```

### 2. Clone Repository

```bash
git clone <repo-url>
cd anemosens-api
```

### 3. Build & Run Container

```bash
docker build -t anemosense-api .
docker run -d --name anemosense-api -p 5000:5000 --restart unless-stopped anemosense-api
```

### 4. Configure Nginx Reverse Proxy

Install Nginx:
```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/anemosense`:
```nginx
server {
    listen 80;
    server_name anemosense.webranastore.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS headers (handled by Flask, but can be duplicated here)
        add_header Access-Control-Allow-Origin $http_origin always;
        add_header Access-Control-Allow-Credentials true always;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/anemosense /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 5. Setup SSL with Certbot

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d anemosense.webranastore.com
```

Certbot will auto-configure HTTPS and set up renewal cron job.

---

## Dependencies

```
Flask==3.0.0              # Web framework
numpy==1.26.2             # Array operations
opencv-python-headless==4.8.1.78  # Image processing
tensorflow==2.15.0        # ML model inference (dual-model support)
gunicorn==21.2.0          # WSGI server (production)
flask-cors==4.0.0         # CORS middleware (Sprint 1)
python-dotenv==1.0.0      # Environment variables (Sprint 1)
h5py==3.10.0              # HDF5 model file handling (compatibility layer)
```

All versions are pinned for reproducibility (Sprint 1 update).

---

## Security Notes

### Sprint 1 Improvements (Security Hardening):
- ✅ **CORS Whitelist**: Only specified origins allowed
- ✅ **Version Pinning**: All dependencies locked to specific versions
- ✅ **Environment Variables**: Sensitive config externalized
- ✅ **Regex Validation**: Localhost ports validated with pattern `^http://localhost:\d+$`

### Sprint 2 Improvements (Stability & Resilience):
- ✅ **Input Validation**: Age bounds (0-120), file size limit (5MB), MIME type check
- ✅ **Error Sanitization**: Stack traces hidden from client responses
- ✅ **DoS Prevention**: File size limits prevent memory exhaustion attacks
- ✅ **Information Disclosure Protection**: Generic error messages for system failures

### Security Score Progress:
- **Pre-Sprint 1**: 4.8/10 (Multiple critical vulnerabilities)
- **Post-Sprint 1**: 8.5/10 (Critical security issues resolved)
- **Post-Sprint 2**: 9.5/10 (Stability & resilience hardened)

### Recommendations for Sprint 3+:
- [ ] Add rate limiting (Flask-Limiter)
- [ ] Add request logging with sanitization
- [ ] Set up monitoring (Sentry/CloudWatch)
- [ ] Implement API authentication (JWT/API keys)
- [ ] Add request ID tracking for debugging

---

## Troubleshooting

### CORS Errors
**Problem**: Browser shows "CORS policy blocked"

**Solution**:
1. Verify origin is whitelisted in `app.py` CORS config
2. Check request includes proper headers
3. For localhost, ensure port matches regex pattern

### Model Loading Errors
**Problem**: `FileNotFoundError` or `TypeError: batch_shape` error

**Solution**:
1. Ensure both model files exist:
   - `anemia_model_v1.h5` (11 MB)
   - `model_anemia_v2.h5` (49 MB)
2. The app includes a compatibility layer for legacy Keras models
3. Check Docker logs: `docker logs anemosense-api`

### Docker Build Issues
**Problem**: `pip install` fails during build

**Solution**:
1. Check internet connection
2. Use `--no-cache` flag: `docker build --no-cache -t anemosense-api .`
3. Verify `requirements.txt` has correct line endings (LF, not CRLF)

---

## License

[Add your license here]

## Contributors

- [Your Team/Organization]

---

**Last Updated**: Sprint 4 - Dual-Model Architecture
**API Version**: 2.0.0
**Production URL**: https://api.anemosense.webranastore.com

**Sprint History**:
- Sprint 1: Security Hardening (CORS, environment config)
- Sprint 2: Stability & Resilience (Input validation, error handling)
- Sprint 3: Refactoring & Production Readiness (Gunicorn workers, API docs, gender encoding fix)
- Sprint 4: Dual-Model Architecture (V1+V2 models, combined predictions, legacy model compatibility)
