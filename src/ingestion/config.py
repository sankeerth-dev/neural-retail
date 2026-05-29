"""Configuration constants for the NeuralRetail ingestion pipeline.

All path constants reference Delta Lake locations. The BRONZE layer stores
raw ingested data partitioned by date. Paths can be overridden via environment
variables for cloud deployments.
"""

import os

# ---------------------------------------------------------------------------
# Bronze Delta Lake paths
# ---------------------------------------------------------------------------
_BRONZE_ROOT = os.environ.get("NR_BRONZE_ROOT", "data/bronze")

BRONZE_BASE: str = _BRONZE_ROOT
BRONZE_POS: str = f"{_BRONZE_ROOT}/pos_transactions"
BRONZE_ECOMMERCE: str = f"{_BRONZE_ROOT}/ecommerce_events"
BRONZE_ERP: str = f"{_BRONZE_ROOT}/erp_inventory"
BRONZE_EXTERNAL: str = f"{_BRONZE_ROOT}/external_signals"

# ---------------------------------------------------------------------------
# Silver Delta Lake paths
# ---------------------------------------------------------------------------
_SILVER_ROOT = os.environ.get("NR_SILVER_ROOT", "data/silver")

SILVER_CUSTOMER_FEATURES: str = f"{_SILVER_ROOT}/customer_features"
SILVER_SKU_DEMAND_FEATURES: str = f"{_SILVER_ROOT}/sku_demand_features"

# ---------------------------------------------------------------------------
# Gold Delta Lake paths
# ---------------------------------------------------------------------------
_GOLD_ROOT = os.environ.get("NR_GOLD_ROOT", "data/gold")

GOLD_FORECAST_OUTPUT: str = f"{_GOLD_ROOT}/forecast_output"
GOLD_CHURN_SCORES: str = f"{_GOLD_ROOT}/churn_scores"

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------
MARQUEZ_URL: str = os.environ.get("NR_MARQUEZ_URL", "http://localhost:5000")
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://localhost:5000"
)

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
GE_THRESHOLD: float = float(os.environ.get("NR_GE_THRESHOLD", "0.98"))

# ---------------------------------------------------------------------------
# Spark configuration
# ---------------------------------------------------------------------------
SPARK_APP_NAME: str = "NeuralRetail-Ingestion"
DELTA_PACKAGE: str = "io.delta:delta-spark_2.12:3.0.0"
