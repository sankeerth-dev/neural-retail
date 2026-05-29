FROM python:3.12-slim

LABEL maintainer="Amdox Technologies <ds-team@amdox.com>"
LABEL description="Streamlit dashboard for NeuralRetail"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
