"""NeuralRetail — Prometheus Metrics Exporter.

Day 20 — NeuralRetail AMX-DS-2026-04
Defines Prometheus Gauges for model KPIs, PSI drift, and API latency.
Pushes metrics to Prometheus Pushgateway every 5 minutes.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus availability check
# ---------------------------------------------------------------------------
try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed; metrics export disabled.")


class NeuralRetailMetricsExporter:
    """Prometheus metrics exporter for NeuralRetail ML model KPIs.

    Exports the following Gauges to Prometheus Pushgateway:
    - ``neuralretail_demand_mape`` (labels: sku_tier) — Demand forecast MAPE.
    - ``neuralretail_churn_auc`` — Churn model AUC-ROC.
    - ``neuralretail_psi`` (labels: feature_name) — Feature PSI drift score.
    - ``neuralretail_stockout_rate`` — Current stockout rate (%).
    - ``neuralretail_api_p95_latency_seconds`` — API P95 latency.

    Usage::

        exporter = NeuralRetailMetricsExporter(pushgateway_url="localhost:9091")
        exporter.push_metrics({"demand_mape_A": 8.7, "churn_auc": 0.921})
        exporter.schedule_push(interval_seconds=300)

    Args:
        pushgateway_url: Prometheus Pushgateway host:port (default localhost:9091).
        job_name: Pushgateway job label (default "neuralretail").
    """

    def __init__(
        self,
        pushgateway_url: str = "localhost:9091",
        job_name: str = "neuralretail",
    ) -> None:
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self._registry = CollectorRegistry() if _PROMETHEUS_AVAILABLE else None
        self._timer: threading.Timer | None = None

        if _PROMETHEUS_AVAILABLE:
            self._init_gauges()
        else:
            self._demand_mape_gauge = None
            self._churn_auc_gauge = None
            self._psi_gauge = None
            self._stockout_rate_gauge = None
            self._api_p95_latency_gauge = None

    def _init_gauges(self) -> None:
        """Initialise all Prometheus Gauge metrics with their label sets."""
        reg = self._registry

        self._demand_mape_gauge = Gauge(
            "neuralretail_demand_mape",
            "Demand forecast MAPE (%) by SKU tier",
            labelnames=["sku_tier"],
            registry=reg,
        )
        self._churn_auc_gauge = Gauge(
            "neuralretail_churn_auc",
            "Churn prediction AUC-ROC score",
            registry=reg,
        )
        self._psi_gauge = Gauge(
            "neuralretail_psi",
            "Feature Population Stability Index (PSI) drift score",
            labelnames=["feature_name"],
            registry=reg,
        )
        self._stockout_rate_gauge = Gauge(
            "neuralretail_stockout_rate",
            "Current stockout rate across all SKUs (%)",
            registry=reg,
        )
        self._api_p95_latency_gauge = Gauge(
            "neuralretail_api_p95_latency_seconds",
            "API P95 latency in seconds",
            registry=reg,
        )
        logger.info("Prometheus gauges initialised.")

    # ------------------------------------------------------------------
    # Metric push
    # ------------------------------------------------------------------

    def push_metrics(self, metrics_dict: dict[str, Any]) -> None:
        """Update all gauges and push to Prometheus Pushgateway.

        Expected keys in ``metrics_dict`` (all optional):

        - ``demand_mape_{tier}`` (e.g. ``demand_mape_A``): float MAPE per tier.
        - ``churn_auc``: float AUC-ROC.
        - ``psi_{feature}`` (e.g. ``psi_recency_days``): float PSI per feature.
        - ``stockout_rate``: float percentage.
        - ``api_p95_latency_seconds``: float seconds.

        Args:
            metrics_dict: Metric name → value mapping. Unknown keys are ignored.
        """
        if not _PROMETHEUS_AVAILABLE:
            logger.debug("Prometheus unavailable; skipping push. Metrics: %s", metrics_dict)
            return

        for key, value in metrics_dict.items():
            try:
                if key.startswith("demand_mape_"):
                    tier = key.split("demand_mape_", 1)[1]
                    self._demand_mape_gauge.labels(sku_tier=tier).set(float(value))

                elif key == "churn_auc":
                    self._churn_auc_gauge.set(float(value))

                elif key.startswith("psi_"):
                    feature_name = key.split("psi_", 1)[1]
                    self._psi_gauge.labels(feature_name=feature_name).set(float(value))

                elif key == "stockout_rate":
                    self._stockout_rate_gauge.set(float(value))

                elif key == "api_p95_latency_seconds":
                    self._api_p95_latency_gauge.set(float(value))

                else:
                    logger.debug("Unknown metric key: '%s'. Skipping.", key)

            except Exception as exc:
                logger.warning("Failed to set gauge for '%s': %s", key, exc)

        # Push to gateway
        try:
            push_to_gateway(
                gateway=self.pushgateway_url,
                job=self.job_name,
                registry=self._registry,
            )
            logger.info("Metrics pushed to Pushgateway at %s.", self.pushgateway_url)
        except Exception as exc:
            logger.error("Failed to push metrics to Pushgateway: %s", exc)

    # ------------------------------------------------------------------
    # Scheduled push
    # ------------------------------------------------------------------

    def schedule_push(
        self,
        interval_seconds: int = 300,
        metrics_fn: Any | None = None,
    ) -> None:
        """Schedule recurring metric pushes using a background daemon thread.

        Schedules :meth:`push_metrics` to run every ``interval_seconds`` seconds
        using a chain of ``threading.Timer`` instances. The timer thread is
        daemonic and will not prevent process exit.

        Args:
            interval_seconds: Push interval in seconds (default 300 = 5 min).
            metrics_fn: Optional callable returning a metrics dict. If None,
                pushes an empty dict (useful for testing the push mechanism).
        """
        def _push_and_reschedule() -> None:
            try:
                metrics = metrics_fn() if callable(metrics_fn) else {}
                self.push_metrics(metrics)
            except Exception as exc:
                logger.error("Scheduled metrics push failed: %s", exc)
            finally:
                # Reschedule
                self._timer = threading.Timer(interval_seconds, _push_and_reschedule)
                self._timer.daemon = True
                self._timer.start()

        self._timer = threading.Timer(interval_seconds, _push_and_reschedule)
        self._timer.daemon = True
        self._timer.start()
        logger.info(
            "Scheduled Prometheus push every %ds to %s.", interval_seconds, self.pushgateway_url
        )

    def stop_scheduled_push(self) -> None:
        """Cancel any pending scheduled push timer.

        Safe to call even if no timer is active.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            logger.info("Scheduled Prometheus push cancelled.")

    # ------------------------------------------------------------------
    # Convenience: push from drift monitor output
    # ------------------------------------------------------------------

    def push_drift_metrics(
        self,
        psi_summary: dict[str, float],
        demand_mape: float | None = None,
        churn_auc: float | None = None,
        stockout_rate: float | None = None,
        api_p95_latency: float | None = None,
    ) -> None:
        """Push a complete set of drift and performance metrics in one call.

        Args:
            psi_summary: Dict of ``{feature: psi}`` from :meth:`DriftMonitor.extract_psi_summary`.
            demand_mape: Current demand MAPE (%). If None, not pushed.
            churn_auc: Current churn AUC-ROC. If None, not pushed.
            stockout_rate: Current stockout rate (%). If None, not pushed.
            api_p95_latency: API P95 latency in seconds. If None, not pushed.
        """
        metrics: dict[str, Any] = {}

        # PSI per feature
        for feat, psi in psi_summary.items():
            if not feat.startswith("_"):
                metrics[f"psi_{feat}"] = psi

        # Model KPIs
        if demand_mape is not None:
            metrics["demand_mape_A"] = demand_mape  # Default to tier A
        if churn_auc is not None:
            metrics["churn_auc"] = churn_auc
        if stockout_rate is not None:
            metrics["stockout_rate"] = stockout_rate
        if api_p95_latency is not None:
            metrics["api_p95_latency_seconds"] = api_p95_latency

        self.push_metrics(metrics)
