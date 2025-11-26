# Production image
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model_anemia.h5 ./

EXPOSE 5000

# Gunicorn Production Configuration:
# -w 4         : 4 workers (formula: 2 * vCPU + 1, asumsi 2 vCPU)
# --threads 2  : 2 threads per worker (untuk I/O blocking saat upload)
# --timeout 120: 120 detik timeout (ML inference bisa lama)
# -b 0.0.0.0   : Bind ke semua interfaces
CMD ["gunicorn", "-w", "4", "--threads", "2", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]
