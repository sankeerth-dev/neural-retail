"""NeuralRetail Scoring API — Pydantic v2 Schemas.

Day 19 — NeuralRetail AMX-DS-2026-04
All request/response schemas for demand forecasting, churn scoring,
customer segmentation, and inventory reorder endpoints.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
_model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Demand schemas
# ---------------------------------------------------------------------------

class DailyForecast(BaseModel):
    """A single-day demand forecast with quantile bounds.

    Attributes:
        date: Calendar date of the forecast.
        p10: 10th-percentile demand (lower confidence bound).
        p50: 50th-percentile demand (median / point forecast).
        p90: 90th-percentile demand (upper confidence bound).
    """

    model_config = _model_config

    date: date = Field(..., description="Forecast date (YYYY-MM-DD).")
    p10: float = Field(..., ge=0.0, description="P10 quantile demand (units).")
    p50: float = Field(..., ge=0.0, description="P50 quantile demand (units).")
    p90: float = Field(..., ge=0.0, description="P90 quantile demand (units).")

    @field_validator("p90")
    @classmethod
    def p90_gte_p50(cls, v: float, info: Any) -> float:
        """Assert P90 ≥ P50 for a valid prediction interval."""
        p50 = info.data.get("p50", 0.0)
        if v < p50:
            raise ValueError(f"p90 ({v}) must be ≥ p50 ({p50}).")
        return v


class DemandRequest(BaseModel):
    """Request body for the demand forecasting endpoint.

    Attributes:
        sku_id: Unique SKU identifier (min length 1).
        store_id: Store identifier. Defaults to "all".
        horizon_days: Number of forecast days (1-90).
        include_confidence: Whether to return p10/p90 bounds.
    """

    model_config = _model_config

    sku_id: str = Field(..., min_length=1, description="SKU identifier.")
    store_id: str = Field(default="all", min_length=1, description="Store identifier.")
    horizon_days: int = Field(default=30, ge=1, le=90, description="Forecast horizon (days).")
    include_confidence: bool = Field(default=True, description="Include P10/P90 quantile bounds.")


class DemandResponse(BaseModel):
    """Response body from the demand forecasting endpoint.

    Attributes:
        sku_id: SKU that was scored.
        store_id: Store filter applied.
        forecasts: List of ``DailyForecast`` objects (one per day).
        mape_expected: Expected MAPE for this SKU tier (from model metadata).
        model_version: MLflow registered model version string.
        cached: Whether this response was served from Redis cache.
        scored_at: UTC timestamp of scoring.
    """

    model_config = _model_config

    sku_id: str
    store_id: str
    forecasts: list[DailyForecast]
    mape_expected: float = Field(..., ge=0.0, description="Expected MAPE (%).")
    model_version: str
    cached: bool = Field(default=False)
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Churn schemas
# ---------------------------------------------------------------------------

class ChurnScore(BaseModel):
    """Churn score for a single customer.

    Attributes:
        customer_id: Customer identifier.
        churn_proba: Churn probability in [0, 1].
        risk_tier: Business-readable risk label.
        top_shap_features: Top driver feature names (if include_shap=True).
    """

    model_config = _model_config

    customer_id: str = Field(..., min_length=1)
    churn_proba: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str = Field(..., description="Critical / High / Medium / Low")
    top_shap_features: list[str] | None = Field(default=None)


class ChurnRequest(BaseModel):
    """Request body for the churn scoring endpoint.

    Attributes:
        customer_ids: List of customer IDs to score (max 1000 per batch).
        include_shap: Whether to return top-5 SHAP features for high-risk customers.
    """

    model_config = _model_config

    customer_ids: list[str] = Field(
        ..., min_length=1, max_length=1000, description="Batch of customer IDs."
    )
    include_shap: bool = Field(default=False, description="Return SHAP features for top-20 high-risk.")

    @field_validator("customer_ids")
    @classmethod
    def customer_ids_not_empty_strings(cls, ids: list[str]) -> list[str]:
        """Strip whitespace and reject blank IDs."""
        cleaned = [cid.strip() for cid in ids if cid.strip()]
        if not cleaned:
            raise ValueError("customer_ids must contain at least one non-empty ID.")
        return cleaned


class ChurnResponse(BaseModel):
    """Response body from the churn scoring endpoint.

    Attributes:
        scores: List of ``ChurnScore`` objects sorted by churn_proba descending.
        model_version: MLflow registered model version.
        scored_at: UTC timestamp.
        total_scored: Total customers scored.
        high_risk_count: Count with churn_proba > 0.6.
    """

    model_config = _model_config

    scores: list[ChurnScore]
    model_version: str
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    total_scored: int
    high_risk_count: int


# ---------------------------------------------------------------------------
# Segment schemas
# ---------------------------------------------------------------------------

class SegmentAssignment(BaseModel):
    """Segment assignment for a single customer.

    Attributes:
        customer_id: Customer identifier.
        cluster_id: Integer cluster assignment (0-indexed).
        persona: Business persona label from cluster profile.
        clv_tier: Customer lifetime value tier (Platinum / Gold / Silver / Bronze).
        recommended_channel: Suggested primary engagement channel.
    """

    model_config = _model_config

    customer_id: str = Field(..., min_length=1)
    cluster_id: int = Field(..., ge=0)
    persona: str
    clv_tier: str = Field(..., description="Platinum / Gold / Silver / Bronze")
    recommended_channel: str = Field(..., description="Email / SMS / Push / In-App")


class SegmentRequest(BaseModel):
    """Request body for the segment scoring endpoint.

    Attributes:
        customer_ids: List of customer IDs to segment (max 5000 per call).
    """

    model_config = _model_config

    customer_ids: list[str] = Field(..., min_length=1, max_length=5000)


class SegmentResponse(BaseModel):
    """Response body from the segment scoring endpoint.

    Attributes:
        assignments: List of ``SegmentAssignment`` objects.
        model_version: KMeans model version from MLflow.
        scored_at: UTC timestamp.
    """

    model_config = _model_config

    assignments: list[SegmentAssignment]
    model_version: str
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Inventory schemas
# ---------------------------------------------------------------------------

class ReorderRec(BaseModel):
    """Reorder recommendation for a single SKU.

    Attributes:
        sku_id: SKU identifier.
        current_stock: Current on-hand inventory (units).
        reorder_point: Trigger level (units) — order when stock hits this.
        recommended_order_qty: EOQ-based recommended purchase order quantity.
        urgency: Critical / High / Medium / OK.
        days_until_stockout: Estimated days before stockout at current demand.
        estimated_po_value: Estimated purchase order value in £.
    """

    model_config = _model_config

    sku_id: str = Field(..., min_length=1)
    current_stock: int = Field(..., ge=0)
    reorder_point: int = Field(..., ge=0)
    recommended_order_qty: int = Field(..., ge=0)
    urgency: str = Field(..., description="Critical / High / Medium / OK")
    days_until_stockout: float = Field(..., ge=0.0)
    estimated_po_value: float = Field(..., ge=0.0, description="£ estimated PO value.")


class ReorderRequest(BaseModel):
    """Request body for the inventory reorder endpoint.

    Attributes:
        sku_ids: List of SKU IDs to evaluate (max 500).
        include_eoq: Whether to compute and return EOQ-based order quantities.
    """

    model_config = _model_config

    sku_ids: list[str] = Field(..., min_length=1, max_length=500)
    include_eoq: bool = Field(default=True, description="Compute EOQ-based recommended quantities.")


class ReorderResponse(BaseModel):
    """Response body from the inventory reorder endpoint.

    Attributes:
        recommendations: List of ``ReorderRec`` objects sorted by urgency.
        total_skus: Total SKUs evaluated.
        critical_count: Count with urgency=Critical.
        scored_at: UTC timestamp.
    """

    model_config = _model_config

    recommendations: list[ReorderRec]
    total_skus: int
    critical_count: int
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response schema.

    Attributes:
        status: "ok" or "degraded".
        models_loaded: Dict of model_name → bool (is loaded in cache).
        timestamp: UTC timestamp.
        version: API version string.
    """

    model_config = _model_config

    status: str
    models_loaded: dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
