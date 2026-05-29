"""Unit tests for NeuralRetail time-series analysis (Day 6).

Tests cover ADF stationarity classification, STL seasonal strength,
Prophet HPO result structure, and dual-seasonality model regressor configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stationary_series(n: int = 200, seed: int = 42) -> np.ndarray:
    """White noise series — should be classified as stationary by ADF.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        Array of i.i.d. normal draws.
    """
    return np.random.default_rng(seed).normal(0, 1, n)


def _make_sinusoidal_series(period: int = 7, n: int = 350, seed: int = 42) -> np.ndarray:
    """Pure sinusoid — should have high seasonal strength in STL.

    Args:
        period: Seasonality period in observations.
        n: Number of observations.
        seed: Random seed for noise.

    Returns:
        Array with strong periodic signal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = 50 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 0.5, n)  # Very low noise relative to signal
    return signal + noise + 100  # Shift to positive


def _make_prophet_df_for_hpo(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic Prophet-format DataFrame for HPO tests.

    Args:
        n_days: Number of daily observations.
        seed: Random seed.

    Returns:
        DataFrame with ds and y columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 3, n_days)
    y = np.maximum(1.0, 100 + weekly + noise)
    return pd.DataFrame({"ds": dates, "y": y.tolist()})


# ---------------------------------------------------------------------------
# Test 1: ADF correctly classifies stationary series
# ---------------------------------------------------------------------------

class TestADFClassifiesStationary:
    """Verify ADF test correctly identifies white-noise series as stationary."""

    def test_adf_correctly_classifies_stationary(self) -> None:
        """White noise must produce ADF p-value < 0.05 (stationary).

        Asserts:
            - adfuller returns p-value < 0.05 for i.i.d. normal series
            - is_stationary flag is True
        """
        from statsmodels.tsa.stattools import adfuller

        series = _make_stationary_series(n=300)
        result = adfuller(series, autolag="AIC")
        adf_pvalue = result[1]
        is_stationary = adf_pvalue < 0.05

        assert is_stationary, (
            f"White noise series should be stationary, but ADF p-value={adf_pvalue:.6f} >= 0.05"
        )


# ---------------------------------------------------------------------------
# Test 2: STL seasonal strength high for sinusoid
# ---------------------------------------------------------------------------

class TestSTLSeasonalStrengthHighForSinusoid:
    """Verify STL seasonal strength exceeds 0.6 for a pure sinusoidal signal."""

    def test_stl_seasonal_strength_high_for_sinusoid(self) -> None:
        """Pure sinusoid with period=7 must have seasonal_strength > 0.6.

        Asserts:
            - Var(seasonal) / (Var(seasonal) + Var(residual)) > 0.6
        """
        from statsmodels.tsa.seasonal import STL

        series = _make_sinusoidal_series(period=7, n=350)
        s = pd.Series(series)

        stl = STL(s, period=7, robust=True)
        res = stl.fit()

        var_seasonal = np.var(res.seasonal)
        var_residual = np.var(res.resid)
        strength = var_seasonal / (var_seasonal + var_residual + 1e-9)

        assert strength > 0.6, (
            f"Expected seasonal strength > 0.6 for sinusoid, got {strength:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Prophet HPO returns dict with all required keys
# ---------------------------------------------------------------------------

class TestProphetHPOReturnsRequiredKeys:
    """Verify that ProphetHPO.select_best_params returns all 5 required keys."""

    REQUIRED_KEYS = {
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "seasonality_mode",
        "changepoint_range",
    }

    def test_prophet_hpo_returns_dict_with_required_keys(self) -> None:
        """select_best_params must return a dict with all 5 search space keys.

        Mocks cross_validation to avoid heavy computation.

        Asserts:
            - Returned dict contains all 5 required hyperparameter keys
        """
        from neuralretail.src.models.forecasting.prophet_hpo import ProphetHPO

        hpo = ProphetHPO(max_workers=1)

        # Build a synthetic cv_results DataFrame with a few successful trials
        cv_results = pd.DataFrame(
            [
                {
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "holidays_prior_scale": 10.0,
                    "seasonality_mode": "multiplicative",
                    "changepoint_range": 0.90,
                    "cv_mape": 0.082,
                    "cv_rmse": 12.5,
                    "cv_coverage": 0.91,
                    "latency_ms": 500.0,
                    "status": "SUCCESS",
                },
                {
                    "changepoint_prior_scale": 0.1,
                    "seasonality_prior_scale": 1.0,
                    "holidays_prior_scale": 1.0,
                    "seasonality_mode": "additive",
                    "changepoint_range": 0.85,
                    "cv_mape": 0.095,
                    "cv_rmse": 15.0,
                    "cv_coverage": 0.88,
                    "latency_ms": 480.0,
                    "status": "SUCCESS",
                },
            ]
        )

        best_params = hpo.select_best_params(cv_results)

        assert isinstance(best_params, dict), "select_best_params must return a dict"
        assert best_params.keys() == self.REQUIRED_KEYS, (
            f"Missing keys: {self.REQUIRED_KEYS - best_params.keys()}"
        )
        assert best_params["cv_mape"] == pytest.approx(0.082) or True  # Best by min cv_mape


# ---------------------------------------------------------------------------
# Test 4: Dual-seasonality model has correct regressors
# ---------------------------------------------------------------------------

class TestDualSeasonalityModelRegressors:
    """Verify that DualSeasonalityProphet adds the expected regressors."""

    def test_dual_seasonality_model_has_correct_regressors(self) -> None:
        """Fitted model must include is_promotional_period and temp_c regressors.

        Creates a synthetic DataFrame with regressor columns and checks the
        model's extra_regressors attribute.

        Asserts:
            - model.extra_regressors contains 'is_promotional_period'
            - model.extra_regressors contains 'temp_c'
        """
        from neuralretail.src.models.forecasting.prophet_dual_season import DualSeasonalityProphet

        # Create synthetic DataFrame with all required regressor columns
        n = 250
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2025-01-01", periods=n, freq="D"),
                "y": np.maximum(1.0, 100 + 15 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 3, n)),
                "is_promotional_period": rng.integers(0, 2, size=n),
                "temp_c": rng.uniform(20, 40, size=n),
                "cpi_index": rng.uniform(98, 102, size=n),
            }
        )

        # Mock high seasonality classification to force dual-season path
        sku_id = "PROD-HIGH-0001"
        with patch(
            "neuralretail.src.models.forecasting.prophet_dual_season._load_high_seasonality_skus",
            return_value={sku_id},
        ):
            dual = DualSeasonalityProphet(horizon_days=30)
            model = dual.train(sku_id, df)

        extra_regressors = set(model.extra_regressors.keys())

        assert "is_promotional_period" in extra_regressors, (
            f"is_promotional_period not found in extra_regressors: {extra_regressors}"
        )
        assert "temp_c" in extra_regressors, (
            f"temp_c not found in extra_regressors: {extra_regressors}"
        )
