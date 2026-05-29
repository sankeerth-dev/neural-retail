"""NeuralRetail Scoring API — Middleware.

Day 19 — NeuralRetail AMX-DS-2026-04
Request timing middleware (Prometheus histogram) and model cache middleware
with MLflow registry version watching.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (optional — graceful if not installed)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Histogram

    REQUEST_DURATION = Histogram(
        "neuralretail_request_duration_seconds",
        "HTTP request duration in seconds",
        labelnames=["endpoint", "method", "status_code"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed; request metrics disabled.")


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records per-request latency to Prometheus.

    Each request is timed and its duration pushed to the
    ``neuralretail_request_duration_seconds`` Histogram with labels:
    ``endpoint``, ``method``, and ``status_code``.

    Latency is also logged at DEBUG level for non-Prometheus environments.
    """

    def __init__(self, app: ASGIApp, exclude_paths: list[str] | None = None) -> None:
        """Initialise middleware.

        Args:
            app: The ASGI application to wrap.
            exclude_paths: List of path prefixes to exclude from timing
                (e.g. ["/health", "/metrics"]). Defaults to ["/metrics"].
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/metrics"]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Time the request and record to Prometheus.

        Args:
            request: Incoming FastAPI/Starlette request object.
            call_next: The next middleware or route handler.

        Returns:
            The response from the downstream handler.
        """
        path = request.url.path
        if any(path.startswith(exc) for exc in self.exclude_paths):
            return await call_next(request)

        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        endpoint = path
        method = request.method
        status_code = str(response.status_code)

        logger.debug("REQUEST %s %s → %s in %.3fs", method, endpoint, status_code, duration)

        if _PROMETHEUS_AVAILABLE:
            REQUEST_DURATION.labels(
                endpoint=endpoint, method=method, status_code=status_code
            ).observe(duration)

        # Add latency header for client-side debugging
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"
        return response


# ---------------------------------------------------------------------------
# Model cache middleware / registry
# ---------------------------------------------------------------------------

# Module-level cache: {model_name: loaded_model_object}
_MODEL_CACHE: dict[str, Any] = {}
# Track known versions to detect registry changes
_MODEL_VERSIONS: dict[str, str] = {}


def load_model_from_mlflow(model_name: str, stage: str = "Production") -> Any:
    """Load a model from MLflow Model Registry into the in-process cache.

    Checks the current Production version in the registry. If the version
    differs from the cached version, reloads the model. Otherwise returns
    the cached instance.

    Args:
        model_name: Registered model name in MLflow (e.g. "demand_ensemble").
        stage: Model stage to load from (default "Production").

    Returns:
        Loaded model object (framework-agnostic — returned as-is from MLflow).
        Returns ``None`` if MLflow is unavailable or the model is not found.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            logger.warning("No %s version found for model '%s'.", stage, model_name)
            return None

        current_version = latest_versions[0].version
        cached_version = _MODEL_VERSIONS.get(model_name)

        if cached_version == current_version and model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        # Load from registry
        model_uri = f"models:/{model_name}/{stage}"
        logger.info("Loading model '%s' v%s from MLflow…", model_name, current_version)
        model = mlflow.pyfunc.load_model(model_uri)
        _MODEL_CACHE[model_name] = model
        _MODEL_VERSIONS[model_name] = current_version
        logger.info("Model '%s' v%s loaded into cache.", model_name, current_version)
        return model

    except Exception as exc:
        logger.error("Failed to load model '%s' from MLflow: %s", model_name, exc)
        return None


def get_cached_model(model_name: str) -> Any | None:
    """Retrieve a model from the in-process cache without hitting MLflow.

    Args:
        model_name: Registered model name.

    Returns:
        Cached model object, or ``None`` if not in cache.
    """
    return _MODEL_CACHE.get(model_name)


def preload_all_models(model_names: list[str], stage: str = "Production") -> dict[str, bool]:
    """Pre-load multiple models into cache at application startup.

    Called from the FastAPI lifespan context manager.

    Args:
        model_names: List of registered model names to load.
        stage: Model stage to load from.

    Returns:
        Dict of ``{model_name: True/False}`` indicating load success.
    """
    results: dict[str, bool] = {}
    for name in model_names:
        model = load_model_from_mlflow(name, stage=stage)
        results[name] = model is not None
    return results


def clear_model_cache() -> None:
    """Clear all models from the in-process cache.

    Called from the FastAPI lifespan shutdown handler to release resources.
    """
    _MODEL_CACHE.clear()
    _MODEL_VERSIONS.clear()
    logger.info("Model cache cleared.")


class ModelCacheMiddleware(BaseHTTPMiddleware):
    """Middleware stub that could trigger background cache refresh.

    In production this would poll the MLflow registry on a timer and
    transparently swap model versions without a restart. For Week 3,
    version refresh is handled per-request in :func:`load_model_from_mlflow`.

    This class is kept as a hook for future async refresh logic.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Pass-through dispatch (no-op for now).

        Args:
            request: Incoming request.
            call_next: Next handler.

        Returns:
            Response unchanged.
        """
        return await call_next(request)
