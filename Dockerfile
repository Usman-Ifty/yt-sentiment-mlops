# ── Stage 1: base ──────────────────────────────────────────────────────────
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ─────────────────────────────────────────────────────────────
# Copy only the essentials for the API
COPY models/bert/best_model  /app/models/bert/best_model
COPY data/processed/label_map.json /app/data/processed/label_map.json
COPY src/ /app/src/

# ── Runtime ─────────────────────────────────────────────────────────────────
EXPOSE 5000

ENV PORT=5000

# Run with Gunicorn for production
# RUN pip install gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]

# Run with flask for now
CMD ["python", "src/api/app.py"]
