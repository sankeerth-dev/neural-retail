"""NeuralRetail Scoring API — Churn Prediction Router.

Day 19 — NeuralRetail AMX-DS-2026-04
POST /churn endpoint: batch churn scoring with optional SHAP for
the top-20 highest-risk customers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from src.serving.api.auth import verify_api_key
from src.serving.api.middleware import get_cached_model, load_model_from_mlflow
from src.serving.api.schemas import (
    ChurnRequest,
    ChurnResponse,
    ChurnScore,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Risk tier mapping
# ---------------------------------------------------------------------------
_RISK_TIER_THRESHOLDS = [
    (0.80, "Critical"),
    (0.60, "High"),
    (0.40, "Medium"),
    (0.00, "Low"),
]


def _assign_risk_tier(proba: float) -> str:
    """Map churn probability to a business risk tier label.

    Args:
        proba: Churn probability in [0, 1].

    Returns:
        Risk tier string: Critical / High / Medium / Low.
    """
    for threshold, label in _RISK_TIER_THRESHOLDS:
        if proba >= threshold:
            return label
    return "Low"


# ---------------------------------------------------------------------------
# Mock feature retrieval
# ---------------------------------------------------------------------------

def _get_online_features(customer_ids: list[str]) -> dict[str, dict[str, float]]:
    """Retrieve online features from Feast Redis store.

    In production this calls ``feast.FeatureStore.get_online_features()``.
    Here we simulate plausible RFM + behavioural features.

    Args:
        customer_ids: List of customer identifiers.

    Returns:
        Dict of ``{customer_id: {feature_name: value}}``.
    """
    rng = np.random.default_rng(42)
    result: dict[str, dict[str, float]] = {}
    feature_names = [
        "recency_days", "frequency", "monetary", "avg_basket_size",
        "rfm_score", "rolling_mean_7d", "lag_1d", "day_of_week",
        "is_weekend", "temp_c", "cpi_index", "days_to_next_holiday",
    ]
    for cid in customer_ids:
        seed = hash(cid) % 2**31
        r = np.random.default_rng(seed)
        result[cid] = {
            "recency_days": float(r.integers(1, 180)),
            "frequency": float(r.integers(1, 50)),
            "monetary": float(r.uniform(20, 5000)),
            "avg_basket_size": float(r.uniform(10, 200)),
            "rfm_score": float(r.uniform(1, 5)),
            "rolling_mean_7d": float(r.uniform(10, 200)),
            "lag_1d": float(r.uniform(0, 300)),
            "day_of_week": float(r.integers(0, 6)),
            "is_weekend": float(r.integers(0, 1)),
            "temp_c": float(r.uniform(-5, 35)),
            "cpi_index": float(r.uniform(95, 115)),
            "days_to_next_holiday": float(r.integers(1, 90)),
        }
    return result


def _run_churn_inference(
    customer_ids: list[str],
    feature_map: dict[str, dict[str, float]],
    include_shap: bool,
) -> list[ChurnScore]:
    """Run churn stacking ensemble inference for all customers.

    Args:
        customer_ids: Customer IDs to score.
        feature_map: Features per customer from online feature store.
        include_shap: Whether to compute SHAP for top-20 high-risk customers.

    Returns:
        List of :class:`ChurnScore` objects sorted by churn_proba descending.
    """
    model = get_cached_model("churn_stacking_ensemble")
    if model is None:
        model = load_model_from_mlflow("churn_stacking_ensemble", stage="Production")

    scores: list[ChurnScore] = []
    for cid in customer_ids:
        feat = feature_map.get(cid, {})
        seed = hash(cid) % 2**31
        r = np.random.default_rng(seed)

        # Simulate model predict_proba (replace with real model call)
        recency_norm = min(feat.get("recency_days", 30) / 180.0, 1.0)
        freq_norm = 1.0 - min(feat.get("frequency", 10) / 50.0, 1.0)
        monetary_norm = 1.0 - min(feat.get("monetary", 100) / 5000.0, 1.0)
        base_proba = 0.35 * recency_norm + 0.30 * freq_norm + 0.20 * monetary_norm + 0.15 * float(r.uniform(0, 0.3))
        churn_proba = float(np.clip(base_proba + r.normal(0, 0.04), 0.02, 0.97))

        top_shap: list[str] | None = None
        scores.append(
            ChurnScore(
                customer_id=cid,
                churn_proba=round(churn_proba, 4),
                risk_tier=_assign_risk_tier(churn_proba),
                top_shap_features=top_shap,
            )
        )

    # Sort by churn_proba descending
    scores.sort(key=lambda s: s.churn_proba, reverse=True)

    # Optionally compute SHAP for top-20 high-risk customers only
    if include_shap:
        _add_shap_features(scores[:20], feature_map)

    return scores


def _add_shap_features(
    high_risk_scores: list[ChurnScore],
    feature_map: dict[str, dict[str, float]],
) -> None:
    """Add top-5 SHAP feature names to high-risk customer scores (in-place).

    In production this calls a pre-loaded SHAP TreeExplainer. Here we
    return deterministically ordered feature names by importance.

    Args:
        high_risk_scores: ChurnScore objects to augment (mutated in-place).
        feature_map: Feature values per customer.
    """
    _TOP_FEATURES_BY_ABS_SHAP = [
        "recency_days", "frequency", "monetary",
        "rolling_mean_7d", "days_to_next_holiday",
    ]
    for score in high_risk_scores:
        feat = feature_map.get(score.customer_id, {})
        # Simulate per-customer ordering based on feature values
        r = np.random.default_rng(hash(score.customer_id) % 2**31)
        ordered = sorted(
            _TOP_FEATURES_BY_ABS_SHAP,
            key=lambda f: abs(feat.get(f, 0.0)) + float(r.uniform(0, 0.1)),
            reverse=True,
        )
        score.top_shap_features = ordered[:5]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/churn",
    response_model=ChurnResponse,
    summary="Batch churn probability scoring",
    description=(
        "Score up to 1000 customers per request. Returns probabilities sorted "
        "by risk descending. Set include_shap=true to get top-5 SHAP features "
        "for the highest-risk 20 customers (performance trade-off)."
    ),
)
async def predict_churn(
    request: ChurnRequest,
    api_key: str = Depends(verify_api_key),
) -> ChurnResponse:
    """Score churn probability for a batch of customers.

    Args:
        request: Validated :class:`ChurnRequest` with customer_ids list.
        api_key: Verified API key.

    Returns:
        :class:`ChurnResponse` with scores sorted by churn_proba descending.

    Raises:
        HTTPException: 503 if inference fails.
    """
    if len(request.customer_ids) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="customer_ids must not be empty.",
        )

    try:
        feature_map = _get_online_features(request.customer_ids)
        scores = _run_churn_inference(request.customer_ids, feature_map, request.include_shap)
    except Exception as exc:
        logger.error("Churn inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Churn inference error: {exc}",
        )

    high_risk_count = sum(1 for s in scores if s.churn_proba > 0.6)

    return ChurnResponse(
        scores=scores,
        model_version="v5 (churn_stacking_ensemble)",
        scored_at=datetime.utcnow(),
        total_scored=len(scores),
        high_risk_count=high_risk_count,
    )
