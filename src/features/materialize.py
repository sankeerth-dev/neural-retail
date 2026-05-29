"""Feast feature store materialization and retrieval utilities.

Provides online materialization, online feature retrieval, and historical
(point-in-time) feature retrieval for NeuralRetail models.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from feast import FeatureStore

from neuralretail.src.ingestion.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

FEATURE_STORE_REPO_PATH = "configs"

ALL_FEATURE_REFS = [
    "customer_rfm_fv:recency_days",
    "customer_rfm_fv:frequency",
    "customer_rfm_fv:monetary",
    "customer_rfm_fv:avg_basket_size",
    "customer_rfm_fv:rfm_score",
    "sku_demand_fv:rolling_mean_7d",
    "sku_demand_fv:rolling_mean_14d",
    "sku_demand_fv:rolling_mean_30d",
    "sku_demand_fv:lag_1d",
    "sku_demand_fv:lag_7d",
    "sku_demand_fv:momentum_7d",
    "sku_date_fv:day_of_week",
    "sku_date_fv:week_of_year",
    "sku_date_fv:is_weekend",
    "sku_date_fv:is_promotional_period",
    "sku_date_fv:days_to_next_holiday",
    "sku_external_fv:temp_c",
    "sku_external_fv:rain_mm",
    "sku_external_fv:is_extreme_weather",
    "sku_external_fv:cpi_index",
    "sku_external_fv:cpi_mom_change",
]


def _get_feature_store() -> FeatureStore:
    """Instantiate the Feast FeatureStore from the repo path.

    Returns:
        Feast FeatureStore configured from configs/feature_store.yaml.
    """
    return FeatureStore(repo_path=FEATURE_STORE_REPO_PATH)


def materialize_online(
    feature_store: FeatureStore,
    start_date: datetime,
    end_date: datetime,
) -> dict[str, Any]:
    """Materialize all feature views into the Redis online store.

    Calls feast materialize for the specified date range across all four
    production feature views. Logs materialization metrics to MLflow.

    Args:
        feature_store: Initialized Feast FeatureStore instance.
        start_date: Start of materialization window (inclusive).
        end_date: End of materialization window (inclusive).

    Returns:
        Dict with keys: materialization_rows (int), latency_ms (float),
        feature_freshness (str, ISO datetime of end_date).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    t0 = time.perf_counter()
    feature_views_to_materialize = [
        "customer_rfm_fv",
        "sku_demand_fv",
        "sku_date_fv",
        "sku_external_fv",
    ]

    logger.info(
        "Starting online materialization: %s → %s",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    feature_store.materialize(
        start_date=start_date,
        end_date=end_date,
        feature_views=feature_views_to_materialize,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    freshness_ts = end_date.isoformat()

    logger.info(
        "Materialization complete. Latency=%.0fms freshness=%s",
        latency_ms,
        freshness_ts,
    )

    # Approximate row count — Feast does not expose this directly
    materialization_rows = -1  # Set to actual row count in production via Delta

    with mlflow.start_run(run_name="feast_materialization", nested=True):
        mlflow.log_metrics(
            {
                "materialization_latency_ms": latency_ms,
                "materialization_rows": float(materialization_rows),
            }
        )
        mlflow.log_param("feature_freshness", freshness_ts)
        mlflow.log_param("feature_views", ",".join(feature_views_to_materialize))

    return {
        "materialization_rows": materialization_rows,
        "latency_ms": latency_ms,
        "feature_freshness": freshness_ts,
    }


def get_online_features(
    customer_ids: list[str],
    product_ids: list[str],
    store_ids: list[str],
) -> dict[str, Any]:
    """Retrieve online features for a set of entity IDs from Redis.

    Args:
        customer_ids: List of customer_id strings.
        product_ids: List of product_id strings (same length as customer_ids).
        store_ids: List of store_id strings (same length as customer_ids).

    Returns:
        Dict keyed by entity ID tuple (customer_id, product_id, store_id) with
        feature values from Redis. Also includes a top-level "raw" key with
        the full Feast response dict.
    """
    fs = _get_feature_store()

    entity_rows = [
        {"customer_id": c, "product_id": p, "store_id": s}
        for c, p, s in zip(customer_ids, product_ids, store_ids)
    ]

    feature_refs = ALL_FEATURE_REFS

    t0 = time.perf_counter()
    online_response = fs.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    response_dict = online_response.to_dict()
    logger.info(
        "Online features retrieved: %d entities in %.0fms",
        len(entity_rows),
        latency_ms,
    )

    # Build keyed result
    result: dict[str, Any] = {}
    for i, row in enumerate(entity_rows):
        key = (row["customer_id"], row["product_id"], row["store_id"])
        result[key] = {
            feat: response_dict[feat][i]
            for feat in response_dict
            if feat not in ("customer_id", "product_id", "store_id")
        }

    result["raw"] = response_dict
    return result


def get_historical_features(
    entity_df: pd.DataFrame,
    feature_refs: list[str] | None = None,
) -> pd.DataFrame:
    """Retrieve historical features with point-in-time correct joins.

    Performs a point-in-time correct join between the entity DataFrame and
    the offline feature store. Prevents future data leakage by aligning
    feature timestamps to the entity event_timestamp.

    Args:
        entity_df: DataFrame with entity ID columns and an
            ``event_timestamp`` column (datetime). Required columns vary by
            feature view (e.g., customer_id, product_id).
        feature_refs: List of feature references in "view_name:feature_name"
            format. Defaults to ALL_FEATURE_REFS if None.

    Returns:
        DataFrame with entity columns, event_timestamp, and all requested
        feature columns. Rows with timestamps in the future relative to
        event_timestamp are excluded by Feast PIT logic.
    """
    if feature_refs is None:
        feature_refs = ALL_FEATURE_REFS

    fs = _get_feature_store()

    logger.info(
        "Fetching historical features: %d entity rows, %d feature refs",
        len(entity_df),
        len(feature_refs),
    )

    t0 = time.perf_counter()
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Historical features retrieved: shape=%s in %.0fms",
        training_df.shape,
        latency_ms,
    )
    return training_df
