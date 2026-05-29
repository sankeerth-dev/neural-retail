"""Prophet hyperparameter optimisation (HPO) for NeuralRetail demand forecasting.

Performs grid-search cross-validation over the Prophet hyperparameter search
space using parallel ProcessPoolExecutor, logs all trial results and the best
parameters to MLflow with nested runs.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "demand_forecasting"
MAX_WORKERS = 4


def _get_experiment_id() -> str:
    """Get or create the demand_forecasting MLflow experiment.

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


def _run_single_trial(
    df: pd.DataFrame,
    params: dict[str, Any],
    horizon: str,
    period: str,
    initial: str,
) -> dict[str, Any]:
    """Run one Prophet cross-validation trial with the given parameters.

    Designed to be called from ProcessPoolExecutor (must be top-level for pickling).

    Args:
        df: Prophet-format DataFrame with ds and y columns.
        params: Hyperparameter dict for this trial.
        horizon: CV horizon string (e.g., "30 days").
        period: CV period string (e.g., "7 days").
        initial: CV initial training window string (e.g., "180 days").

    Returns:
        Dict with all params plus cv_mape, cv_rmse, cv_coverage, and latency_ms.
    """
    t0 = time.perf_counter()
    try:
        model = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            changepoint_range=params["changepoint_range"],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        if "is_promotional_period" in df.columns:
            model.add_regressor("is_promotional_period")

        model.fit(df)

        cv_df = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=None,
        )
        perf = performance_metrics(cv_df)

        result = {
            **params,
            "cv_mape": float(perf["mape"].mean()),
            "cv_rmse": float(perf["rmse"].mean()),
            "cv_coverage": float(perf["coverage"].mean()) if "coverage" in perf else 0.9,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "status": "SUCCESS",
        }
    except Exception as exc:
        result = {
            **params,
            "cv_mape": float("inf"),
            "cv_rmse": float("inf"),
            "cv_coverage": 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "status": f"FAILED: {exc}",
        }
    return result


class ProphetHPO:
    """Grid-search hyperparameter optimiser for Facebook Prophet.

    Runs cross-validation across the full search space using parallel workers
    and logs all trial results to MLflow as nested child runs.

    Example:
        >>> hpo = ProphetHPO()
        >>> cv_results = hpo.run_cv_grid_search(sku_df, horizon="30 days")
        >>> best_params = hpo.select_best_params(cv_results)
    """

    SEARCH_SPACE: dict[str, list[Any]] = {
        "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_range": [0.80, 0.85, 0.90, 0.95],
    }

    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        """Initialise ProphetHPO.

        Args:
            max_workers: Number of parallel worker processes for CV.
        """
        self.max_workers = max_workers
        self._exp_id: str = _get_experiment_id()

    def _generate_param_grid(self) -> list[dict[str, Any]]:
        """Generate the full Cartesian product of hyperparameter combinations.

        Returns:
            List of parameter dicts, one per trial.
        """
        keys = list(self.SEARCH_SPACE.keys())
        values = list(self.SEARCH_SPACE.values())
        combinations = list(itertools.product(*values))
        logger.info(
            "HPO search space: %d total combinations across %d parameters",
            len(combinations),
            len(keys),
        )
        return [dict(zip(keys, combo)) for combo in combinations]

    def run_cv_grid_search(
        self,
        df: pd.DataFrame,
        horizon: str = "30 days",
        period: str = "7 days",
        initial: str = "180 days",
    ) -> pd.DataFrame:
        """Run parallel grid-search cross-validation across the search space.

        Uses ProcessPoolExecutor with max_workers parallel processes. Falls back
        to sequential execution if parallelism fails.

        Args:
            df: Prophet-format DataFrame with ds and y columns.
            horizon: Forecast horizon for CV (e.g., "30 days").
            period: Cutoff stride for CV (e.g., "7 days").
            initial: Minimum training window (e.g., "180 days").

        Returns:
            DataFrame with one row per trial containing all hyperparameters
            plus cv_mape, cv_rmse, cv_coverage, latency_ms, and status.
        """
        param_grid = self._generate_param_grid()
        logger.info("Starting HPO grid search: %d trials, max_workers=%d", len(param_grid), self.max_workers)

        results = []
        t_total = time.perf_counter()

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(_run_single_trial, df, p, horizon, period, initial): p
                    for p in param_grid
                }
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results.append(result)
                    if i % 10 == 0:
                        logger.info("HPO progress: %d/%d trials complete", i, len(param_grid))
        except Exception as exc:
            logger.warning("Parallel HPO failed (%s) — falling back to sequential", exc)
            for params in param_grid:
                result = _run_single_trial(df, params, horizon, period, initial)
                results.append(result)

        elapsed = time.perf_counter() - t_total
        logger.info(
            "HPO complete: %d/%d trials succeeded in %.1fs",
            sum(1 for r in results if r["status"] == "SUCCESS"),
            len(results),
            elapsed,
        )

        return pd.DataFrame(results)

    def select_best_params(self, cv_results: pd.DataFrame) -> dict[str, Any]:
        """Select the best hyperparameters by minimum cv_mape.

        Args:
            cv_results: DataFrame from run_cv_grid_search().

        Returns:
            Dict of best hyperparameter values for all 5 search dimensions.
        """
        successful = cv_results[cv_results["status"] == "SUCCESS"]
        if successful.empty:
            logger.error("No successful HPO trials — returning default parameters")
            return {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "seasonality_mode": "multiplicative",
                "changepoint_range": 0.90,
            }

        best_idx = successful["cv_mape"].idxmin()
        best_row = successful.loc[best_idx]
        best_params = {
            "changepoint_prior_scale": best_row["changepoint_prior_scale"],
            "seasonality_prior_scale": best_row["seasonality_prior_scale"],
            "holidays_prior_scale": best_row["holidays_prior_scale"],
            "seasonality_mode": best_row["seasonality_mode"],
            "changepoint_range": best_row["changepoint_range"],
        }

        logger.info(
            "Best HPO params: MAPE=%.4f params=%s",
            best_row["cv_mape"],
            best_params,
        )
        return best_params

    def log_hpo_results(
        self,
        cv_results: pd.DataFrame,
        best_params: dict[str, Any],
        experiment_id: str,
    ) -> None:
        """Log HPO results to MLflow with nested child runs per trial.

        The parent run captures the best parameters and summary metrics.
        Each trial is logged as a nested child run.

        Args:
            cv_results: Full grid-search results DataFrame.
            best_params: Best parameter dict from select_best_params().
            experiment_id: MLflow experiment ID for the demand forecasting experiment.
        """
        artifact_path = Path("artifacts/hpo")
        artifact_path.mkdir(parents=True, exist_ok=True)
        csv_path = str(artifact_path / "cv_results.csv")
        cv_results.to_csv(csv_path, index=False)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="prophet_hpo_grid_search",
        ) as parent_run:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            successful = cv_results[cv_results["status"] == "SUCCESS"]
            if not successful.empty:
                mlflow.log_metrics(
                    {
                        "best_cv_mape": float(successful["cv_mape"].min()),
                        "mean_cv_mape": float(successful["cv_mape"].mean()),
                        "n_trials": len(cv_results),
                        "n_successful": len(successful),
                    }
                )
            mlflow.log_artifact(csv_path, artifact_path="hpo")
            mlflow.set_tags({"hpo_type": "grid_search", "model": "prophet", "project": "neuralretail"})

            # Log each trial as a child run (limit to top-20 by MAPE for brevity)
            top_trials = successful.nsmallest(min(20, len(successful)), "cv_mape")
            for _, row in top_trials.iterrows():
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"trial_mape_{row['cv_mape']:.4f}",
                    nested=True,
                ):
                    trial_params = {
                        k: row[k]
                        for k in [
                            "changepoint_prior_scale",
                            "seasonality_prior_scale",
                            "holidays_prior_scale",
                            "seasonality_mode",
                            "changepoint_range",
                        ]
                    }
                    mlflow.log_params(trial_params)
                    mlflow.log_metrics(
                        {
                            "cv_mape": float(row["cv_mape"]),
                            "cv_rmse": float(row["cv_rmse"]),
                            "cv_coverage": float(row["cv_coverage"]),
                        }
                    )

            logger.info(
                "HPO results logged to MLflow parent_run=%s", parent_run.info.run_id
            )
