"""Feast feature store definitions for NeuralRetail.

Defines entities, feature views (offline + online), and on-demand feature
views for customer RFM, SKU demand, SKU calendar, and external signal features.
This module is the single source of truth for the Feast feature registry.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
    RequestSource,
)
from feast.types import Bool, Float64, Int64, String

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

customer_features_source = FileSource(
    name="customer_features_source",
    path="data/silver/customer_features.parquet",
    timestamp_field="snapshot_date",
    description="Silver-layer customer RFM features written daily",
)

sku_demand_features_source = FileSource(
    name="sku_demand_features_source",
    path="data/silver/sku_demand_features.parquet",
    timestamp_field="date",
    description="Silver-layer SKU demand time-series features",
)

sku_external_features_source = FileSource(
    name="sku_external_features_source",
    path="data/silver/sku_external_features.parquet",
    timestamp_field="date",
    description="Silver-layer external signals joined to SKU-store level",
)

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=String,
    description="Retail customer — POS + e-commerce unified identity",
)

product = Entity(
    name="product",
    join_keys=["product_id"],
    value_type=String,
    description="Retail product / SKU identifier",
)

store = Entity(
    name="store",
    join_keys=["store_id"],
    value_type=String,
    description="Physical retail store identifier",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

customer_rfm_fv = FeatureView(
    name="customer_rfm_fv",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="recency_days", dtype=Int64, description="Days since last purchase"),
        Field(name="frequency", dtype=Int64, description="Transaction count in 90-day window"),
        Field(name="monetary", dtype=Float64, description="Total spend in 90-day window (INR)"),
        Field(name="avg_basket_size", dtype=Float64, description="Average transaction value"),
        Field(name="rfm_score", dtype=Float64, description="Composite RFM score (1–5)"),
    ],
    source=customer_features_source,
    description="Customer RFM features computed daily with 90-day lookback",
    tags={"team": "ds", "project": "neuralretail", "layer": "silver"},
    online=True,
)

sku_demand_fv = FeatureView(
    name="sku_demand_fv",
    entities=[product],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_mean_7d", dtype=Float64, description="7-day rolling average demand"),
        Field(name="rolling_mean_14d", dtype=Float64, description="14-day rolling average demand"),
        Field(name="rolling_mean_30d", dtype=Float64, description="30-day rolling average demand"),
        Field(name="lag_1d", dtype=Float64, description="Demand lag 1 day"),
        Field(name="lag_7d", dtype=Float64, description="Demand lag 7 days"),
        Field(name="momentum_7d", dtype=Float64, description="7-day momentum vs rolling mean"),
    ],
    source=sku_demand_features_source,
    description="SKU demand time-series features with rolling windows and lags",
    tags={"team": "ds", "project": "neuralretail", "layer": "silver"},
    online=True,
)

sku_date_fv = FeatureView(
    name="sku_date_fv",
    entities=[product],
    ttl=timedelta(days=1),
    schema=[
        Field(name="day_of_week", dtype=Int64, description="Day of week (0=Mon, 6=Sun)"),
        Field(name="week_of_year", dtype=Int64, description="ISO week of year"),
        Field(name="is_weekend", dtype=Bool, description="True if Saturday or Sunday"),
        Field(
            name="is_promotional_period",
            dtype=Bool,
            description="True if within 7 days of a major retail holiday",
        ),
        Field(
            name="days_to_next_holiday",
            dtype=Int64,
            description="Days until next retail holiday",
        ),
    ],
    source=sku_demand_features_source,
    description="Calendar and promotional date features at SKU granularity",
    tags={"team": "ds", "project": "neuralretail", "layer": "silver"},
    online=True,
)

sku_external_fv = FeatureView(
    name="sku_external_fv",
    entities=[product, store],
    ttl=timedelta(days=1),
    schema=[
        Field(name="temp_c", dtype=Float64, description="Daily mean temperature (°C)"),
        Field(name="rain_mm", dtype=Float64, description="Daily total precipitation (mm)"),
        Field(
            name="is_extreme_weather",
            dtype=Bool,
            description="True if temp>40°C or rain>50mm",
        ),
        Field(name="cpi_index", dtype=Float64, description="Consumer price index for category"),
        Field(
            name="cpi_mom_change",
            dtype=Float64,
            description="Month-over-month CPI change (%)",
        ),
    ],
    source=sku_external_features_source,
    description="External signals: weather and CPI joined at SKU-store-date level",
    tags={"team": "ds", "project": "neuralretail", "layer": "silver"},
    online=True,
)

# ---------------------------------------------------------------------------
# On-demand feature view — churn risk signals
# ---------------------------------------------------------------------------

customer_rfm_request_source = RequestSource(
    name="customer_rfm_request_source",
    schema=[
        Field(name="rfm_score", dtype=Float64),
        Field(name="recency_days", dtype=Int64),
        Field(name="monetary", dtype=Float64),
    ],
)


@OnDemandFeatureView(
    sources=[customer_rfm_fv],
    schema=[
        Field(
            name="high_risk_flag",
            dtype=Bool,
            description="True if rfm_score<2.0 and recency_days>60",
        ),
        Field(
            name="clv_tier",
            dtype=String,
            description="Customer lifetime value tier: high/medium/low",
        ),
    ],
    description="On-demand churn risk signals derived from RFM features",
)
def churn_risk_odfv(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute on-demand churn risk features from RFM inputs.

    Args:
        inputs: DataFrame with rfm_score (float), recency_days (int),
            monetary (float) columns.

    Returns:
        DataFrame with high_risk_flag (bool) and clv_tier (str) columns.
    """
    df = pd.DataFrame()
    df["high_risk_flag"] = (inputs["rfm_score"] < 2.0) & (inputs["recency_days"] > 60)
    df["clv_tier"] = inputs["monetary"].apply(
        lambda m: "high" if m > 5000 else ("medium" if m > 1000 else "low")
    )
    return df


# ---------------------------------------------------------------------------
# Feature services
# ---------------------------------------------------------------------------

demand_forecasting_fs = FeatureService(
    name="demand_forecasting_service",
    features=[sku_demand_fv, sku_date_fv, sku_external_fv],
    description="All features needed for demand forecasting models",
)

churn_prediction_fs = FeatureService(
    name="churn_prediction_service",
    features=[customer_rfm_fv, churn_risk_odfv],
    description="All features needed for churn prediction models",
)
