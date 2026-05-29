"""Setup validation script for NeuralRetail Week 1.

Checks all 7 required services and prints a health-check score.
"""

import sys
import time
from typing import Any

import psycopg2
import redis
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "neuralretail"
PG_USER = "nr_user"
PG_PASS = "nr_pass"

REDIS_HOST = "localhost"
REDIS_PORT = 6379

MLFLOW_URL = "http://localhost:5000/health"
AIRFLOW_URL = "http://localhost:8080/health"
STREAMLIT_URL = "http://localhost:8501"

BRONZE_BASE = "data/bronze"
FEAST_FEATURE_STORE_PATH = "configs/feature_store.yaml"


def _check_postgres() -> tuple[bool, str]:
    """Check PostgreSQL connectivity.

    Returns:
        Tuple of (success, message).
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASS,
            connect_timeout=5,
        )
        conn.close()
        return True, "PostgreSQL connected successfully"
    except Exception as exc:
        return False, f"PostgreSQL FAIL: {exc}"


def _check_redis() -> tuple[bool, str]:
    """Check Redis connectivity.

    Returns:
        Tuple of (success, message).
    """
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=5)
        r.ping()
        return True, "Redis ping OK"
    except Exception as exc:
        return False, f"Redis FAIL: {exc}"


def _check_http(name: str, url: str) -> tuple[bool, str]:
    """Check an HTTP endpoint.

    Args:
        name: Human-readable service name.
        url: URL to GET.

    Returns:
        Tuple of (success, message).
    """
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code < 500:
            return True, f"{name} responded {resp.status_code}"
        return False, f"{name} FAIL: HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"{name} FAIL: {exc}"


def _check_delta_lake() -> tuple[bool, str]:
    """Check Delta Lake by attempting to read the bronze base path.

    Returns:
        Tuple of (success, message).
    """
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("validate_setup")
            .config(
                "spark.jars.packages",
                "io.delta:delta-spark_2.12:3.0.0",
            )
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .master("local[2]")
            .getOrCreate()
        )
        spark.read.format("delta").load(BRONZE_BASE)
        return True, "Delta Lake bronze path readable"
    except Exception as exc:
        if "Path does not exist" in str(exc) or "Unable to infer schema" in str(exc):
            return True, "Delta Lake libraries OK (bronze path empty — expected)"
        return False, f"Delta Lake FAIL: {exc}"


def _check_feast() -> tuple[bool, str]:
    """Check Feast feature store initialization.

    Returns:
        Tuple of (success, message).
    """
    try:
        import os

        if not os.path.exists(FEAST_FEATURE_STORE_PATH):
            return (
                True,
                "Feast check skipped — feature_store.yaml not yet created (expected Day 4)",
            )
        from feast import FeatureStore

        store = FeatureStore(repo_path="configs")
        fvs = store.list_feature_views()
        return True, f"Feast OK — {len(fvs)} feature views registered"
    except Exception as exc:
        return False, f"Feast FAIL: {exc}"


def run_validation() -> None:
    """Run all service health checks and print a summary score."""
    checks: list[dict[str, Any]] = [
        {"name": "PostgreSQL", "fn": _check_postgres},
        {"name": "Redis", "fn": _check_redis},
        {"name": "MLflow", "fn": lambda: _check_http("MLflow", MLFLOW_URL)},
        {"name": "Airflow", "fn": lambda: _check_http("Airflow", AIRFLOW_URL)},
        {"name": "Streamlit", "fn": lambda: _check_http("Streamlit", STREAMLIT_URL)},
        {"name": "Delta Lake", "fn": _check_delta_lake},
        {"name": "Feast", "fn": _check_feast},
    ]

    print("\n" + "=" * 60)
    print("  NeuralRetail — Week 1 Service Validation")
    print("=" * 60)

    passed = 0
    for check in checks:
        t0 = time.perf_counter()
        ok, msg = check["fn"]()
        elapsed = (time.perf_counter() - t0) * 1000
        status = "✅ OK  " if ok else "❌ FAIL"
        print(f"  {status}  {check['name']:<12}  {msg}  ({elapsed:.0f}ms)")
        if ok:
            passed += 1

    print("=" * 60)
    print(f"  SCORE: {passed}/7 services healthy")
    print("=" * 60 + "\n")

    if passed < 7:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
