"""NeuralRetail Scoring API — FastAPI Application.

Day 19 — NeuralRetail AMX-DS-2026-04
Main FastAPI app with lifespan model loading, middleware registration,
router inclusion, health check, and Prometheus metrics exposition.

Targets:
    - P95 API latency < 1.5 seconds.
    - Redis cache TTL 3600s to avoid redundant inference.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response

from src.serving.api.middleware import (
    ModelCacheMiddleware,
    RequestTimingMiddleware,
    clear_model_cache,
    get_cached_model,
    preload_all_models,
)
from src.serving.api.routers import demand as demand_router_module
from src.serving.api.routers import churn as churn_router_module
from src.serving.api.routers import segment as segment_router_module
from src.serving.api.routers import inventory as inventory_router_module
from src.serving.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registered model names to pre-load
# ---------------------------------------------------------------------------
_PRODUCTION_MODELS = [
    "demand_ensemble",
    "churn_stacking_ensemble",
    "kmeans_segmentation",
    "price_elasticity_electronics",
]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler for startup/shutdown model management.

    On startup: pre-load all Production models from MLflow into the
    module-level cache. Logs success/failure for each model.

    On shutdown: clear model cache and release resources.

    Args:
        app: FastAPI application instance.

    Yields:
        None (context manager body is the running application).
    """
    logger.info("NeuralRetail API starting up — pre-loading models…")
    load_results = preload_all_models(_PRODUCTION_MODELS, stage="Production")
    for model_name, success in load_results.items():
        if success:
            logger.info("✓ Model '%s' loaded into cache.", model_name)
        else:
            logger.warning("✗ Model '%s' not available — will attempt lazy load on demand.", model_name)

    yield  # Application runs here

    logger.info("NeuralRetail API shutting down — releasing model cache…")
    clear_model_cache()
    logger.info("Model cache cleared. Shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the NeuralRetail FastAPI application.

    Returns:
        Fully configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title="NeuralRetail Scoring API",
        description=(
            "Production ML scoring API for demand forecasting, churn prediction, "
            "customer segmentation, and inventory intelligence. "
            "Project AMX-DS-2026-04."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ──────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        RequestTimingMiddleware,
        exclude_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )
    app.add_middleware(ModelCacheMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(
        demand_router_module.router,
        prefix="/api/v1/predict",
        tags=["Demand Forecasting"],
    )
    app.include_router(
        churn_router_module.router,
        prefix="/api/v1/predict",
        tags=["Churn Prediction"],
    )
    app.include_router(
        segment_router_module.router,
        prefix="/api/v1/segment",
        tags=["Customer Segmentation"],
    )
    app.include_router(
        inventory_router_module.router,
        prefix="/api/v1/inventory",
        tags=["Inventory Intelligence"],
    )

    return app


# Create the module-level app instance
app = create_app()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Platform"],
    summary="API health check",
    description="Returns API status and model cache state. No authentication required.",
)
async def health_check() -> HealthResponse:
    """Return API health status and per-model load state.

    Returns:
        :class:`~src.serving.api.schemas.HealthResponse` with status,
        models_loaded dict, and timestamp.
    """
    models_loaded = {name: get_cached_model(name) is not None for name in _PRODUCTION_MODELS}
    status_str = "ok" if all(models_loaded.values()) else "degraded"
    if not any(models_loaded.values()):
        status_str = "unhealthy"

    return HealthResponse(
        status=status_str,
        models_loaded=models_loaded,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    tags=["Platform"],
    summary="Prometheus metrics",
    description="Expose Prometheus metrics in text format. Scrape this endpoint with your Prometheus instance.",
    include_in_schema=False,
)
async def prometheus_metrics() -> Response:
    """Return Prometheus metrics in the standard text exposition format.

    Returns:
        Plain-text Prometheus metrics response, or empty if prometheus_client
        is not installed.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        logger.warning("prometheus_client not installed; /metrics returns empty.")
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint — redirects to docs."""
    return {
        "message": "NeuralRetail Scoring API v1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
