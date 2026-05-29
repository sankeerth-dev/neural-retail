"""NeuralRetail Scoring API — Inventory Reorder Router.

Day 19 — NeuralRetail AMX-DS-2026-04
POST /reorder endpoint: per-SKU reorder recommendations with EOQ-based
order quantities, urgency classification, and estimated PO value.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from src.serving.api.auth import verify_api_key
from src.serving.api.middleware import get_cached_model
from src.serving.api.schemas import (
    ReorderRec,
    ReorderRequest,
    ReorderResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_UNIT_COST = 25.0  # £ per unit
_DEFAULT_ORDER_COST = 150.0  # £ per purchase order
_DEFAULT_HOLDING_PCT = 20.0  # % of unit cost per year
_DEFAULT_LEAD_TIME = 14  # days
_SAFETY_FACTOR = 1.65  # 95% service level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_eoq(
    annual_demand: float,
    order_cost: float = _DEFAULT_ORDER_COST,
    holding_cost_pct: float = _DEFAULT_HOLDING_PCT,
    unit_cost: float = _DEFAULT_UNIT_COST,
) -> int:
    """Compute Wilson EOQ for given demand parameters.

    Args:
        annual_demand: Annual demand in units.
        order_cost: Fixed ordering cost per purchase order (£).
        holding_cost_pct: Annual holding cost as % of unit cost.
        unit_cost: Per-unit cost (£).

    Returns:
        EOQ rounded to nearest integer unit.
    """
    H = (holding_cost_pct / 100.0) * unit_cost
    if H <= 0 or annual_demand <= 0:
        return 0
    return max(1, round(math.sqrt((2.0 * annual_demand * order_cost) / H)))


def _days_to_urgency(days: float) -> str:
    """Map days-until-stockout to urgency label.

    Args:
        days: Estimated days until stockout.

    Returns:
        Urgency string: Critical / High / Medium / OK.
    """
    if days <= 7:
        return "Critical"
    elif days <= 14:
        return "High"
    elif days <= 30:
        return "Medium"
    return "OK"


def _simulate_sku_inventory(sku_id: str) -> dict:
    """Simulate realistic inventory state for a SKU.

    In production this queries the inventory management system.

    Args:
        sku_id: SKU identifier.

    Returns:
        Dict with current_stock, avg_daily_demand, reorder_point,
        unit_cost, lead_time_days.
    """
    seed = hash(sku_id) % 2**31
    r = np.random.default_rng(seed)
    avg_daily = float(r.uniform(5, 80))
    lead_time = int(r.integers(7, 28))
    safety_stock = _SAFETY_FACTOR * math.sqrt(avg_daily) * math.sqrt(lead_time)
    reorder_point = avg_daily * lead_time + safety_stock
    return {
        "current_stock": int(r.integers(0, 500)),
        "avg_daily_demand": avg_daily,
        "annual_demand": avg_daily * 365,
        "reorder_point": round(reorder_point),
        "unit_cost": round(float(r.uniform(5, 200)), 2),
        "lead_time_days": lead_time,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/reorder",
    response_model=ReorderResponse,
    summary="SKU reorder recommendations",
    description=(
        "Compute reorder recommendations for up to 500 SKUs. "
        "Returns EOQ-based order quantities, urgency classification, "
        "and estimated purchase order values."
    ),
)
async def get_reorder_recommendations(
    request: ReorderRequest,
    api_key: str = Depends(verify_api_key),
) -> ReorderResponse:
    """Generate reorder recommendations for a batch of SKUs.

    For each SKU:
    1. Retrieves current inventory state (stock, demand, reorder point).
    2. Computes days_until_stockout = current_stock / avg_daily_demand.
    3. Computes EOQ if include_eoq=True.
    4. Assigns urgency tier.
    5. Estimates purchase order value = order_qty * unit_cost.

    Args:
        request: Validated :class:`ReorderRequest` with sku_ids list.
        api_key: Verified API key.

    Returns:
        :class:`ReorderResponse` sorted by urgency (Critical first).

    Raises:
        HTTPException: 503 on unexpected inference error.
    """
    try:
        recommendations: list[ReorderRec] = []
        for sku_id in request.sku_ids:
            inv = _simulate_sku_inventory(sku_id)
            avg_daily = max(inv["avg_daily_demand"], 0.1)
            current_stock = inv["current_stock"]
            days_until_stockout = round(current_stock / avg_daily, 1)
            urgency = _days_to_urgency(days_until_stockout)

            if request.include_eoq:
                order_qty = _compute_eoq(
                    annual_demand=inv["annual_demand"],
                    unit_cost=inv["unit_cost"],
                )
            else:
                # Simple coverage rule: order enough for 30 days
                order_qty = max(0, round(avg_daily * 30 - current_stock))

            estimated_po_value = round(order_qty * inv["unit_cost"], 2)

            recommendations.append(
                ReorderRec(
                    sku_id=sku_id,
                    current_stock=current_stock,
                    reorder_point=inv["reorder_point"],
                    recommended_order_qty=order_qty,
                    urgency=urgency,
                    days_until_stockout=days_until_stockout,
                    estimated_po_value=estimated_po_value,
                )
            )

        # Sort: Critical first, then by days ascending
        urgency_rank = {"Critical": 0, "High": 1, "Medium": 2, "OK": 3}
        recommendations.sort(
            key=lambda r: (urgency_rank.get(r.urgency, 4), r.days_until_stockout)
        )

        critical_count = sum(1 for r in recommendations if r.urgency == "Critical")

    except Exception as exc:
        logger.error("Inventory reorder inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inventory inference error: {exc}",
        )

    return ReorderResponse(
        recommendations=recommendations,
        total_skus=len(recommendations),
        critical_count=critical_count,
        scored_at=datetime.utcnow(),
    )
