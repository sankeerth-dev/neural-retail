FROM python:3.12-slim

LABEL maintainer="Amdox Technologies <ds-team@amdox.com>"
LABEL description="MLflow tracking server for NeuralRetail"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mlflow==2.13.*" \
    "psycopg2-binary" \
    "boto3"

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:5000/health || exit 1
