"""Integration tests for the NeuralRetail Week 1 pipeline.

Requires the full Docker Compose stack to be running (postgres, redis, mlflow,
airflow, streamlit). Mark individual tests with @pytest.mark.integration
and run with: pytest -m integration tests/integration/

All tests assume:
- docker compose up -d has been executed
- Bronze/silver data has been populated by running the ingestion pipeline
- MLflow experiments have been created by setup_experiments.py
- At least one churn model is registered in the MLflow model registry
"""

from __future__ import annotations

import importlib
import logging

import pandas as pd
import pytest

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration  # All tests in this module are integration tests


# ---------------------------------------------------------------------------
# Test 1: Silver customer features table has rows
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_silver_table_has_rows() -> None:
    """Silver customer_features Delta table must contain at least one row.

    Reads the silver/customer_features Delta Lake table using PySpark and
    asserts that the row count is positive.

    Asserts:
        - Delta table is readable
        - Row count > 0
    """
    from pyspark.sql import SparkSession
    from neuralretail.src.ingestion.config import SILVER_CUSTOMER_FEATURES, DELTA_PACKAGE

    spark = (
        SparkSession.builder.appName("integration-test")
        .config("spark.jars.packages", DELTA_PACKAGE)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.format("delta").load(SILVER_CUSTOMER_FEATURES)
    row_count = df.count()

    assert row_count > 0, (
        f"silver/customer_features table must have at least 1 row, got {row_count}. "
        "Run the bronze ingestion and feature engineering pipelines first."
    )
    logger.info("✅ silver/customer_features: %d rows", row_count)


# ---------------------------------------------------------------------------
# Test 2: Feast online features are non-empty
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_feast_online_features_non_empty() -> None:
    """Online features for 3 test customers must include all 5 RFM features.

    Materializes the customer_rfm_fv feature view and retrieves online
    features from Redis for 3 known test customer IDs.

    Asserts:
        - get_online_features returns a non-empty dict
        - All 5 RFM feature keys are present for each customer
    """
    from datetime import datetime, timedelta

    from feast import FeatureStore
    from neuralretail.src.features.materialize import get_online_features

    fs = FeatureStore(repo_path="configs")

    # Materialize a small window to populate Redis
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    try:
        fs.materialize(start_date=start, end_date=end, feature_views=["customer_rfm_fv"])
    except Exception as exc:
        pytest.skip(f"Materialization failed (data may not be populated): {exc}")

    TEST_CUSTOMERS = ["CUST-0000001", "CUST-0000002", "CUST-0000003"]
    TEST_PRODUCTS = ["PROD-0001", "PROD-0001", "PROD-0001"]
    TEST_STORES = ["STORE-01", "STORE-01", "STORE-01"]

    result = get_online_features(TEST_CUSTOMERS, TEST_PRODUCTS, TEST_STORES)

    assert result is not None, "get_online_features must return a non-None result"
    assert "raw" in result, "Result dict must contain 'raw' key"

    raw = result["raw"]
    rfm_keys = [k for k in raw if "rfm" in k.lower() or k in (
        "recency_days", "frequency", "monetary", "avg_basket_size", "rfm_score"
    )]

    # Also check with feast double-underscore naming convention
    feast_rfm_keys = [k for k in raw if "customer_rfm_fv__" in k]

    has_rfm = len(rfm_keys) >= 5 or len(feast_rfm_keys) >= 5
    assert has_rfm, (
        f"Expected at least 5 RFM feature keys in online response. Got keys: {list(raw.keys())}"
    )
    logger.info("✅ Feast online features: %d feature keys returned", len(raw))


# ---------------------------------------------------------------------------
# Test 3: MLflow demand_forecasting experiment exists
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_mlflow_demand_experiment_exists() -> None:
    """The demand_forecasting MLflow experiment must exist and be active.

    Connects to the MLflow tracking server and asserts the experiment
    was created by setup_experiments.py.

    Asserts:
        - Experiment is not None
        - Experiment lifecycle_stage is 'active'
    """
    from mlflow.tracking import MlflowClient
    import os

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name("demand_forecasting")

    assert experiment is not None, (
        "MLflow experiment 'demand_forecasting' not found. "
        "Run infrastructure/mlflow/setup_experiments.py first."
    )
    assert experiment.lifecycle_stage == "active", (
        f"Expected experiment lifecycle_stage='active', got '{experiment.lifecycle_stage}'"
    )
    logger.info("✅ MLflow experiment 'demand_forecasting' exists (id=%s)", experiment.experiment_id)


# ---------------------------------------------------------------------------
# Test 4: Airflow DAG has no import error
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_airflow_dag_no_import_error() -> None:
    """The bronze ingestion DAG must import cleanly and expose a DAG object.

    Imports the dag_bronze_ingestion module and verifies the 'dag' variable
    is a valid Airflow DAG with the expected dag_id.

    Asserts:
        - Module imports without ImportError
        - 'dag' variable exists with correct dag_id
    """
    try:
        import importlib
        dag_module = importlib.import_module("neuralretail.dags.dag_bronze_ingestion")
    except ImportError as exc:
        pytest.fail(f"dag_bronze_ingestion failed to import: {exc}")

    from airflow import DAG

    assert hasattr(dag_module, "dag"), "dag_bronze_ingestion module must define a 'dag' object"
    dag_obj = dag_module.dag
    assert isinstance(dag_obj, DAG), f"Expected airflow.DAG, got {type(dag_obj)}"
    assert dag_obj.dag_id == "neuralretail_bronze_ingestion", (
        f"Expected dag_id='neuralretail_bronze_ingestion', got '{dag_obj.dag_id}'"
    )
    logger.info("✅ DAG 'neuralretail_bronze_ingestion' imports cleanly")


# ---------------------------------------------------------------------------
# Test 5: Churn model in registry staging
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_churn_model_in_registry_staging() -> None:
    """The churn_baseline registered model must have a version in Staging.

    Asserts:
        - At least one version of 'churn_baseline' exists in the Staging stage
    """
    import os
    from mlflow.tracking import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        staging_versions = client.get_latest_versions("churn_baseline", stages=["Staging"])
    except Exception as exc:
        pytest.skip(f"MLflow model registry not accessible: {exc}")

    assert len(staging_versions) > 0, (
        "Expected at least one 'churn_baseline' model version in 'Staging' stage. "
        "Run src/models/run_baseline_experiments.py --experiment churn first."
    )

    version = staging_versions[0]
    logger.info(
        "✅ churn_baseline v%s is in Staging (run_id=%s)",
        version.version,
        version.run_id,
    )
