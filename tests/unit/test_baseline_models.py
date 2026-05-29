"""Unit tests for NeuralRetail Week 1 baseline models.

Covers Prophet training, churn label construction, MLflow parameter logging,
and model registry staging.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prophet_df(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic Prophet-format DataFrame.

    Args:
        n_days: Number of daily observations.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ds (datetime) and y (float) columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base = 100.0
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 5, n_days)
    y = np.maximum(1.0, base + weekly + noise)
    return pd.DataFrame({"ds": dates, "y": y, "is_promotional_period": 0})


def _make_rfm_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic customer RFM DataFrame.

    Args:
        n: Number of customers.
        seed: Random seed.

    Returns:
        DataFrame with RFM columns and churn-triggering rows.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "recency_days": rng.integers(1, 180, size=n),
            "frequency": rng.integers(1, 20, size=n),
            "monetary": rng.uniform(100, 10_000, size=n).round(2),
            "avg_basket_size": rng.uniform(50, 2_000, size=n).round(2),
            "rfm_score": rng.uniform(1.0, 5.0, size=n).round(3),
            "snapshot_date": pd.Timestamp("2026-05-01"),
        }
    )
    # Inject known churners
    df.loc[0:99, "recency_days"] = 60
    df.loc[0:99, "frequency"] = 1
    return df


# ---------------------------------------------------------------------------
# Test 1: Prophet trains without error
# ---------------------------------------------------------------------------

class TestProphetTrainsWithoutError:
    """Verify that Prophet model training completes on 6 months of data."""

    def test_prophet_trains_without_error(self) -> None:
        """Prophet must train successfully on 180 days of synthetic demand.

        Asserts:
            - No exception is raised during train()
            - Returned model has a predict() method
        """
        from neuralretail.src.models.forecasting.baseline_prophet import BaselineProphetForecaster

        forecaster = BaselineProphetForecaster(horizon_days=30)
        df = _make_prophet_df(n_days=200)

        model = forecaster.train("PROD-0001", df)

        assert model is not None, "Expected a fitted Prophet model"
        assert hasattr(model, "predict"), "Model must have a predict() method"

        # Verify model can generate a forecast
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        assert "yhat" in forecast.columns, "Forecast must contain 'yhat' column"
        assert len(forecast) > 0, "Forecast must be non-empty"


# ---------------------------------------------------------------------------
# Test 2: Churn label construction
# ---------------------------------------------------------------------------

class TestChurnFeaturesLoadCorrectLabel:
    """Verify that churn labels are correctly assigned based on RFM heuristic."""

    def test_churn_features_load_correct_label(self) -> None:
        """Rows with recency_days > 30 AND frequency < 2 must have label=1.

        Asserts:
            - Customers with recency_days=60 and frequency=1 get label=1
            - Customers with low recency and high frequency get label=0
        """
        from neuralretail.src.models.churn.baseline_logistic import BaselineChurnClassifier

        clf = BaselineChurnClassifier()
        df = _make_rfm_df(n=500)

        with patch("pandas.read_parquet", return_value=df):
            X, y = clf.load_features("dummy_path")

        # First 100 rows have recency=60 and frequency=1 → should be churned
        churned_idx = df[(df["recency_days"] > 30) & (df["frequency"] < 2)].index
        assert len(churned_idx) > 0, "Expected at least some churned customers"

        # Verify label assignment
        expected_churn = ((df["recency_days"] > 30) & (df["frequency"] < 2)).astype(int)
        assert y.reset_index(drop=True).equals(
            expected_churn.reset_index(drop=True)
        ), "Churn labels do not match expected heuristic"


# ---------------------------------------------------------------------------
# Test 3: MLflow parameter logging
# ---------------------------------------------------------------------------

class TestMLflowParamsAllLogged:
    """Verify that all required parameters are logged to MLflow."""

    REQUIRED_DEMAND_PARAMS = {
        "sku_id",
        "horizon_days",
        "seasonality_mode",
        "changepoint_prior_scale",
    }

    def test_mlflow_params_all_logged(self) -> None:
        """log_to_mlflow must call mlflow.log_params with all required keys.

        Uses a mock mlflow context to capture log_params calls.

        Asserts:
            - mlflow.log_params is called at least once
            - All required parameter keys are present in the call
        """
        from neuralretail.src.models.forecasting.baseline_prophet import BaselineProphetForecaster

        forecaster = BaselineProphetForecaster(horizon_days=30)

        mock_prophet_model = MagicMock()
        mock_prophet_model.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2026-01-01", periods=37, freq="D")}
        )
        mock_prophet_model.predict.return_value = pd.DataFrame(
            {
                "ds": pd.date_range("2026-01-01", periods=37, freq="D"),
                "yhat": [100.0] * 37,
                "yhat_lower": [90.0] * 37,
                "yhat_upper": [110.0] * 37,
            }
        )

        with patch("mlflow.start_run") as mock_run, \
             patch("mlflow.log_params") as mock_log_params, \
             patch("mlflow.log_metrics"), \
             patch("mlflow.set_tags"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.prophet.log_model"):

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            forecaster.log_to_mlflow(
                sku_id="PROD-0001",
                model=mock_prophet_model,
                metrics={"mape": 0.08, "rmse": 12.5, "mae": 8.1, "pi_coverage_90": 0.91},
                params={"changepoint_prior_scale": 0.05},
                sku_tier="A",
            )

        assert mock_log_params.called, "mlflow.log_params must be called"
        logged_params = mock_log_params.call_args[0][0]
        for required_key in self.REQUIRED_DEMAND_PARAMS:
            assert required_key in logged_params, (
                f"Required param '{required_key}' not found in logged params: {logged_params}"
            )


# ---------------------------------------------------------------------------
# Test 4: Model registered in staging
# ---------------------------------------------------------------------------

class TestModelRegisteredAsStaging:
    """Verify that trained models are registered in the MLflow model registry."""

    def test_model_registered_as_staging(self) -> None:
        """log_to_mlflow must call mlflow.prophet.log_model with registry name.

        Asserts:
            - mlflow.prophet.log_model is called with registered_model_name set
            - registered_model_name equals 'prophet_baseline'
        """
        from neuralretail.src.models.forecasting.baseline_prophet import (
            BaselineProphetForecaster,
            MODEL_REGISTRY_NAME,
        )

        forecaster = BaselineProphetForecaster(horizon_days=30)

        mock_model = MagicMock()
        mock_model.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2026-01-01", periods=37, freq="D")}
        )
        mock_model.predict.return_value = pd.DataFrame(
            {
                "ds": pd.date_range("2026-01-01", periods=37, freq="D"),
                "yhat": [100.0] * 37,
                "yhat_lower": [90.0] * 37,
                "yhat_upper": [110.0] * 37,
            }
        )

        with patch("mlflow.start_run") as mock_run, \
             patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.set_tags"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.prophet.log_model") as mock_log_model:

            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            forecaster.log_to_mlflow(
                sku_id="PROD-9999",
                model=mock_model,
                metrics={"mape": 0.09, "rmse": 10.0, "mae": 7.0, "pi_coverage_90": 0.92},
                params={},
                sku_tier="B",
            )

        assert mock_log_model.called, "mlflow.prophet.log_model must be called"
        call_kwargs = mock_log_model.call_args[1]
        assert call_kwargs.get("registered_model_name") == MODEL_REGISTRY_NAME, (
            f"Expected registered_model_name='{MODEL_REGISTRY_NAME}', "
            f"got '{call_kwargs.get('registered_model_name')}'"
        )
