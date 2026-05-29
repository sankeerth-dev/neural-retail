"""ARIMA benchmark for NeuralRetail demand forecasting.

Fits ARIMA models with recommended (p,d,q) orders from the Day 6 ACF/PACF
analysis and compares them against baseline Prophet MAPE. Uses pmdarima
auto_arima as a fallback when orders are missing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "demand_forecasting"
ARIMA_ORDERS_PATH = Path("configs/arima_orders.json")
ARTIFACT_DIR = Path("artifacts/arima")


def _get_experiment_id() -> str:
    """Retrieve or create the demand_forecasting MLflow experiment.

    Returns:
        Experiment ID string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return mlflow.create_experiment(
            EXPERIMENT_NAME, tags={"team": "ds", "project": "neuralretail"}
        )
    return exp.experiment_id


def _load_arima_orders() -> dict[str, dict[str, int]]:
    """Load recommended ARIMA (p,d,q) orders from the Day 6 config file.

    Returns:
        Dict mapping sku_id to {"p": int, "d": int, "q": int}.
        Returns an empty dict if the file does not exist.
    """
    if not ARIMA_ORDERS_PATH.exists():
        logger.warning("arima_orders.json not found at %s", ARIMA_ORDERS_PATH)
        return {}
    with open(ARIMA_ORDERS_PATH) as f:
        return json.load(f)


def _compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        actual: Array of true values.
        predicted: Array of predicted values.

    Returns:
        MAPE as a float (e.g., 0.12 for 12%).
    """
    denom = np.where(np.abs(actual) < 1e-8, 1.0, np.abs(actual))
    return float(np.mean(np.abs((actual - predicted) / denom)))


class ARIMABenchmark:
    """ARIMA model benchmarking against baseline Prophet for NeuralRetail.

    Fits ARIMA models using statsmodels for SKUs with known (p,d,q) orders,
    falls back to pmdarima auto_arima for others, and compares MAPE against
    the baseline Prophet results.

    Example:
        >>> benchmark = ARIMABenchmark()
        >>> comparison_df = benchmark.run(top_n_skus=20)
    """

    def __init__(self, horizon_days: int = 30) -> None:
        """Initialise the ARIMA benchmark.

        Args:
            horizon_days: Forecast horizon for MAPE evaluation.
        """
        self.horizon_days = horizon_days
        self._exp_id: str = _get_experiment_id()
        self._arima_orders: dict[str, dict[str, int]] = _load_arima_orders()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def _load_sku_data(self) -> pd.DataFrame:
        """Load or synthesise SKU daily demand data.

        Returns:
            DataFrame with columns: product_id, date (or ds), quantity (or y).
        """
        try:
            df = pd.read_parquet("data/silver/sku_demand_features")
            if "quantity" not in df.columns and "y" in df.columns:
                df = df.rename(columns={"y": "quantity"})
            if "date" not in df.columns and "ds" in df.columns:
                df = df.rename(columns={"ds": "date"})
            return df
        except Exception as exc:
            logger.warning("Using synthetic data: %s", exc)
            rng = np.random.default_rng(42)
            n_days = 400
            dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
            records = []
            for i in range(20):
                base = rng.integers(80, 300)
                weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
                noise = rng.normal(0, base * 0.05, n_days)
                y = np.maximum(1, base + weekly + noise)
                for j, d in enumerate(dates):
                    records.append({"product_id": f"PROD-{i:04d}", "date": d, "quantity": float(y[j])})
            return pd.DataFrame(records)

    def _fit_arima(
        self,
        series: np.ndarray,
        order: tuple[int, int, int],
        sku_id: str,
    ) -> tuple[float, str]:
        """Fit a statsmodels ARIMA model and compute holdout MAPE.

        Args:
            series: Time-series array.
            order: (p, d, q) ARIMA order tuple.
            sku_id: SKU identifier for logging.

        Returns:
            Tuple of (mape float, model_description str).
        """
        from statsmodels.tsa.arima.model import ARIMA

        train = series[: -self.horizon_days]
        test = series[-self.horizon_days :]

        model = ARIMA(train, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=self.horizon_days)
        mape = _compute_mape(test, np.array(forecast))
        return mape, f"ARIMA{order}"

    def _fit_auto_arima(
        self, series: np.ndarray, sku_id: str
    ) -> tuple[float, str]:
        """Fit pmdarima auto_arima and compute holdout MAPE.

        Args:
            series: Time-series array.
            sku_id: SKU identifier for logging.

        Returns:
            Tuple of (mape float, model_description str).
        """
        import pmdarima as pm

        train = series[: -self.horizon_days]
        test = series[-self.horizon_days :]

        model = pm.auto_arima(
            train,
            seasonal=True,
            m=7,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        forecast = model.predict(n_periods=self.horizon_days)
        mape = _compute_mape(test, np.array(forecast))
        order = model.order
        seasonal_order = model.seasonal_order
        return mape, f"ARIMA{order}x{seasonal_order}"

    def _fit_prophet_baseline(self, series_df: pd.DataFrame, sku_id: str) -> float:
        """Fit baseline Prophet and compute holdout MAPE for comparison.

        Args:
            series_df: DataFrame with ds and y columns.
            sku_id: SKU identifier.

        Returns:
            Prophet MAPE float.
        """
        from prophet import Prophet

        train_df = series_df.iloc[: -self.horizon_days]
        test_y = series_df["y"].iloc[-self.horizon_days :].values

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        model.fit(train_df)
        future = model.make_future_dataframe(periods=self.horizon_days)
        forecast = model.predict(future)
        pred = forecast["yhat"].tail(self.horizon_days).values
        return _compute_mape(test_y, pred)

    def run(self, top_n_skus: int = 20) -> pd.DataFrame:
        """Run ARIMA benchmark for top-N SKUs and compare against Prophet.

        For each SKU:
        1. Fit ARIMA with (p,d,q) from arima_orders.json if available.
        2. Fit auto_arima regardless (for comparison).
        3. Fit baseline Prophet.
        4. Determine the winner by minimum MAPE.

        Args:
            top_n_skus: Number of top-revenue SKUs to benchmark.

        Returns:
            DataFrame with columns: sku_id, arima_mape, prophet_mape,
            auto_arima_mape, arima_order, winner.
        """
        df = self._load_sku_data()
        top_skus = (
            df.groupby("product_id")["quantity"]
            .sum()
            .nlargest(top_n_skus)
            .index.tolist()
        )

        records = []
        for sku_id in top_skus:
            sku_series = (
                df[df["product_id"] == sku_id]
                .set_index("date")["quantity"]
                .sort_index()
                .dropna()
            )
            if len(sku_series) < self.horizon_days + 60:
                logger.warning("Skipping %s — insufficient data", sku_id)
                continue

            series_arr = sku_series.values
            series_df = pd.DataFrame(
                {"ds": sku_series.index, "y": sku_series.values}
            ).reset_index(drop=True)

            # ARIMA with recommended order
            arima_mape = float("nan")
            arima_desc = "N/A"
            if sku_id in self._arima_orders:
                order_cfg = self._arima_orders[sku_id]
                order = (order_cfg["p"], order_cfg["d"], order_cfg["q"])
                try:
                    arima_mape, arima_desc = self._fit_arima(series_arr, order, sku_id)
                except Exception as exc:
                    logger.warning("ARIMA fit failed for %s: %s", sku_id, exc)

            # Auto ARIMA
            auto_mape = float("nan")
            try:
                auto_mape, auto_desc = self._fit_auto_arima(series_arr, sku_id)
            except Exception as exc:
                logger.warning("Auto ARIMA failed for %s: %s", sku_id, exc)
                auto_desc = "FAILED"

            # Prophet baseline
            prophet_mape = float("nan")
            try:
                prophet_mape = self._fit_prophet_baseline(series_df, sku_id)
            except Exception as exc:
                logger.warning("Prophet baseline failed for %s: %s", sku_id, exc)

            mapes = {
                "arima": arima_mape,
                "prophet": prophet_mape,
                "auto_arima": auto_mape,
            }
            winner = min(mapes, key=lambda k: mapes[k] if not np.isnan(mapes[k]) else float("inf"))

            record = {
                "sku_id": sku_id,
                "arima_mape": round(arima_mape, 4),
                "prophet_mape": round(prophet_mape, 4),
                "auto_arima_mape": round(auto_mape, 4),
                "arima_order": arima_desc,
                "winner": winner,
            }
            records.append(record)
            logger.info(
                "SKU=%s arima=%.4f prophet=%.4f auto_arima=%.4f winner=%s",
                sku_id,
                arima_mape,
                prophet_mape,
                auto_mape,
                winner,
            )

        comparison_df = pd.DataFrame(records)

        # Save and log to MLflow
        csv_path = str(ARTIFACT_DIR / "arima_vs_prophet_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)

        with mlflow.start_run(
            experiment_id=self._exp_id,
            run_name="arima_benchmark",
        ):
            mlflow.log_artifact(csv_path, artifact_path="benchmarks")
            if not comparison_df.empty:
                mlflow.log_metrics(
                    {
                        "arima_avg_mape": float(comparison_df["arima_mape"].mean(skipna=True)),
                        "prophet_avg_mape": float(comparison_df["prophet_mape"].mean(skipna=True)),
                        "auto_arima_avg_mape": float(comparison_df["auto_arima_mape"].mean(skipna=True)),
                    }
                )
            mlflow.set_tags({"task": "arima_vs_prophet_benchmark", "project": "neuralretail"})

        logger.info("ARIMA benchmark complete: %d SKUs processed", len(comparison_df))
        return comparison_df
