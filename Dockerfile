FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only the packages the API actually needs at runtime
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    flask-cors==4.0.0 \
    transformers==4.41.2 \
    safetensors==0.4.3 \
    torch==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip

COPY src/api/app.py         ./src/api/app.py
COPY data/processed/label_map.json ./data/processed/label_map.json
COPY models/bert/best_model ./models/bert/best_model

EXPOSE 5000
CMD ["python", "src/api/app.py"]
