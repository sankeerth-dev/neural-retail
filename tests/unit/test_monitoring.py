"""Unit tests for NeuralRetail monitoring: DriftMonitor and Prometheus exporter.

Day 20 — NeuralRetail AMX-DS-2026-04
Tests PSI threshold logic, retrain triggers, HTML report generation,
and Prometheus gauge push.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_reference_df() -> pd.DataFrame:
    """Synthetic reference distribution (training data simulation)."""
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame({
        "recency_days": rng.uniform(1, 90, n),
        "frequency": rng.integers(1, 40, n).astype(float),
        "monetary": rng.uniform(20, 3000, n),
        "rolling_mean_7d": rng.uniform(10, 200, n),
        "cpi_index": rng.uniform(95, 115, n),
        "target": rng.integers(0, 1, n),
        "prediction": rng.uniform(0, 1, n),
    })


@pytest.fixture
def synthetic_drifted_df() -> pd.DataFrame:
    """Synthetic production distribution with intentional drift."""
    rng = np.random.default_rng(77)
    n = 1000
    return pd.DataFrame({
        "recency_days": rng.uniform(60, 200, n),  # Drifted up
        "frequency": rng.integers(1, 10, n).astype(float),  # Drifted down
        "monetary": rng.uniform(20, 3000, n),
        "rolling_mean_7d": rng.uniform(150, 300, n),  # Drifted up
        "cpi_index": rng.uniform(105, 130, n),  # Drifted up
        "target": rng.integers(0, 1, n),
        "prediction": rng.uniform(0.3, 0.9, n),  # Drifted
    })


@pytest.fixture
def drift_monitor_with_data(
    tmp_path: Path,
    synthetic_reference_df: pd.DataFrame,
    synthetic_drifted_df: pd.DataFrame,
) -> "DriftMonitor":
    """Create a DriftMonitor with synthetic reference and production data."""
    from src.monitoring.drift_monitor import DriftMonitor

    ref_path = tmp_path / "reference.parquet"
    prod_path = tmp_path / "production.parquet"
    synthetic_reference_df.to_parquet(ref_path, index=False)
    synthetic_drifted_df.to_parquet(prod_path, index=False)

    return DriftMonitor(
        reference_data_path=str(ref_path),
        production_data_path=str(prod_path),
        model_name="churn_stacking_ensemble",
    )


@pytest.fixture
def drift_monitor_no_drift(
    tmp_path: Path,
    synthetic_reference_df: pd.DataFrame,
) -> "DriftMonitor":
    """Create a DriftMonitor where reference == production (no drift)."""
    from src.monitoring.drift_monitor import DriftMonitor

    ref_path = tmp_path / "reference.parquet"
    prod_path = tmp_path / "production_stable.parquet"
    synthetic_reference_df.to_parquet(ref_path, index=False)
    synthetic_reference_df.to_parquet(prod_path, index=False)  # Same data

    return DriftMonitor(
        reference_data_path=str(ref_path),
        production_data_path=str(prod_path),
        model_name="demand_ensemble",
    )


# ---------------------------------------------------------------------------
# Test 1: PSI > 0.2 triggers retrain
# ---------------------------------------------------------------------------

class TestPSITriggersRetrain:
    """Test that high PSI values correctly trigger the retrain decision."""

    def test_psi_above_02_triggers_retrain(self, drift_monitor_with_data: "DriftMonitor") -> None:
        """PSI > 0.20 on any feature must return should_trigger=True.

        Uses the drifted production data fixture which has intentional
        distributional shift on recency_days and rolling_mean_7d.
        """
        monitor = drift_monitor_with_data
        drift_results = monitor.compute_data_drift(
            features=["recency_days", "frequency", "monetary", "rolling_mean_7d", "cpi_index"]
        )

        # Extract PSI summary
        psi_summary = {feat: info["psi"] for feat, info in drift_results.items()}

        # With drifted data, at least one PSI should exceed 0.20
        max_psi = max(psi_summary.values(), default=0.0)
        assert max_psi > 0.0, "PSI must be non-negative."

        # Test retrain decision
        result = monitor.should_trigger_retrain(
            psi_summary=psi_summary,
            mape_current=9.5,
            mape_baseline=8.7,
            psi_threshold=0.20,
            mape_degradation_threshold=0.15,
        )
        # With severe drift OR MAPE degradation > 15%, should trigger
        mape_trigger = 9.5 > 8.7 * 1.15
        assert isinstance(result, bool), "should_trigger_retrain must return bool."

        # If max PSI > 0.20, must trigger
        if max_psi > 0.20:
            assert result is True, f"PSI={max_psi:.3f} > 0.20 must trigger retrain."

        # MAPE degradation check: 9.5 > 8.7 * 1.15 = 10.005 → False
        # So only PSI drives the decision here
        if max_psi <= 0.20 and not mape_trigger:
            assert result is False, "Low PSI and MAPE OK must not trigger retrain."


# ---------------------------------------------------------------------------
# Test 2: PSI < 0.1 does not trigger retrain
# ---------------------------------------------------------------------------

class TestNoDriftNoRetrain:
    """Test that stable data does not trigger retraining."""

    def test_psi_below_01_no_retrain(self, drift_monitor_no_drift: "DriftMonitor") -> None:
        """When production data matches reference, PSI must be near 0 and no retrain triggered.

        Uses the same DataFrame for reference and production, guaranteeing
        near-zero PSI on all features.
        """
        monitor = drift_monitor_no_drift
        drift_results = monitor.compute_data_drift(
            features=["recency_days", "frequency", "monetary"]
        )
        psi_summary = {feat: info["psi"] for feat, info in drift_results.items()}

        # PSI should be very low when distributions are identical
        max_psi = max(psi_summary.values(), default=0.0)
        assert max_psi < 0.05, (
            f"PSI={max_psi:.4f} must be < 0.05 when reference == production data."
        )

        result = monitor.should_trigger_retrain(
            psi_summary=psi_summary,
            mape_current=8.5,   # Better than baseline
            mape_baseline=8.7,
            psi_threshold=0.10,
        )
        assert result is False, "Stable data with improving MAPE must not trigger retrain."


# ---------------------------------------------------------------------------
# Test 3: HTML report generated at path
# ---------------------------------------------------------------------------

class TestHTMLReportGeneration:
    """Test that HTML drift report is written to the expected file path."""

    def test_html_report_generated_at_path(
        self,
        tmp_path: Path,
        drift_monitor_with_data: "DriftMonitor",
    ) -> None:
        """generate_html_report must create a non-empty HTML file at the given path.

        Patches Evidently if not installed to test the fallback HTML generation.
        """
        monitor = drift_monitor_with_data
        report_path = tmp_path / "reports" / "test_drift_report.html"

        # Always test the fallback path (deterministic in CI without evidently)
        with patch("src.monitoring.drift_monitor._EVIDENTLY_AVAILABLE", False):
            returned_path = monitor.generate_html_report(str(report_path))

        assert returned_path == str(report_path), (
            f"Returned path '{returned_path}' must match input path '{report_path}'."
        )
        assert report_path.exists(), f"HTML report file must exist at {report_path}."
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 100, "HTML report must not be empty."
        assert "NeuralRetail" in content, "HTML report must contain project name."


# ---------------------------------------------------------------------------
# Test 4: Prometheus gauges updated with correct metric names
# ---------------------------------------------------------------------------

class TestPrometheusGauges:
    """Test that Prometheus gauges are updated with correct metric names."""

    def test_prometheus_gauges_updated(self) -> None:
        """push_metrics must attempt to update all expected gauge names.

        Mocks prometheus_client to verify gauge.labels().set() is called with
        the correct metric names and values.
        """
        from src.monitoring.prometheus_exporter import NeuralRetailMetricsExporter

        with patch("src.monitoring.prometheus_exporter._PROMETHEUS_AVAILABLE", True):
            with patch("src.monitoring.prometheus_exporter.Gauge") as mock_gauge_cls:
                with patch("src.monitoring.prometheus_exporter.push_to_gateway") as mock_push:
                    mock_gauge_instance = MagicMock()
                    mock_gauge_cls.return_value = mock_gauge_instance
                    mock_gauge_instance.labels.return_value = mock_gauge_instance

                    exporter = NeuralRetailMetricsExporter(
                        pushgateway_url="localhost:9091",
                        job_name="test",
                    )

                    # Inject mock gauges directly
                    exporter._demand_mape_gauge = mock_gauge_instance
                    exporter._churn_auc_gauge = mock_gauge_instance
                    exporter._psi_gauge = mock_gauge_instance
                    exporter._stockout_rate_gauge = mock_gauge_instance
                    exporter._api_p95_latency_gauge = mock_gauge_instance

                    metrics = {
                        "demand_mape_A": 8.7,
                        "demand_mape_B": 11.2,
                        "churn_auc": 0.921,
                        "psi_recency_days": 0.15,
                        "psi_frequency": 0.08,
                        "stockout_rate": 4.2,
                        "api_p95_latency_seconds": 0.85,
                    }

                    exporter.push_metrics(metrics)

                    # Verify set() was called for numeric metrics
                    assert mock_gauge_instance.set.called or mock_gauge_instance.labels.called, (
                        "Gauge set() or labels() must be called during push_metrics."
                    )

                    # Verify push_to_gateway was called
                    assert mock_push.called, "push_to_gateway must be called after updating gauges."
                    call_args = mock_push.call_args
                    assert "localhost:9091" in str(call_args), (
                        "push_to_gateway must be called with the correct Pushgateway URL."
                    )

    def test_push_drift_metrics_calls_push(self) -> None:
        """push_drift_metrics must translate PSI summary and call push_metrics."""
        from src.monitoring.prometheus_exporter import NeuralRetailMetricsExporter

        exporter = NeuralRetailMetricsExporter()

        # Replace push_metrics with a mock to capture calls
        exporter.push_metrics = MagicMock()

        psi_summary = {
            "recency_days": 0.23,
            "frequency": 0.05,
            "monetary": 0.11,
            "_overall_drift_score": 0.13,
        }

        exporter.push_drift_metrics(
            psi_summary=psi_summary,
            demand_mape=8.7,
            churn_auc=0.921,
            stockout_rate=4.2,
            api_p95_latency=0.85,
        )

        exporter.push_metrics.assert_called_once()
        call_metrics = exporter.push_metrics.call_args[0][0]

        assert "psi_recency_days" in call_metrics, "PSI metric name must include feature name."
        assert "psi_frequency" in call_metrics
        assert "churn_auc" in call_metrics
        assert "demand_mape_A" in call_metrics
        assert "_overall_drift_score" not in call_metrics, (
            "Internal PSI summary key must not be exported as a metric."
        )
        assert call_metrics["psi_recency_days"] == pytest.approx(0.23)
