"""NeuralRetail — Evidently AI Drift Monitor.

Day 20 — NeuralRetail AMX-DS-2026-04
Computes data drift (PSI, KS), model performance drift, prediction drift,
generates HTML reports, and decides whether to trigger retraining.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional evidently import (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import (
        ClassificationPreset,
        DataDriftPreset,
        DataQualityPreset,
        RegressionPreset,
    )
    from evidently.metrics import (
        DataDriftTable,
    )
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnDrift

    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning("evidently not installed; drift monitoring will use fallback statistics.")


class DriftMonitor:
    """Evidently AI-powered drift monitoring for NeuralRetail ML models.

    Computes:
    - Data drift (PSI per feature, KS statistic per feature).
    - Model performance drift (AUC / MAPE over time).
    - Prediction distribution drift (Jensen-Shannon divergence).
    - Full HTML reports for Evidently dashboard.

    Args:
        reference_data_path: Path to Parquet/CSV reference dataset (training
            distribution). Used as the baseline for all drift comparisons.
        production_data_path: Path to Parquet/CSV of recent production
            scoring data (last 7 days by default).
        model_name: MLflow model name for report naming and labelling.
    """

    def __init__(
        self,
        reference_data_path: str | Path,
        production_data_path: str | Path,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.reference_data_path = Path(reference_data_path)
        self.production_data_path = Path(production_data_path)

        self.reference_data: pd.DataFrame = self._load_data(self.reference_data_path)
        self.production_data: pd.DataFrame = self._load_data(self.production_data_path)

        logger.info(
            "DriftMonitor initialised: model=%s  ref=%d rows  prod=%d rows",
            model_name,
            len(self.reference_data),
            len(self.production_data),
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, path: Path) -> pd.DataFrame:
        """Load reference or production dataset from Parquet or CSV.

        Generates a synthetic dataset for environments where the files
        don't exist yet (e.g., development / CI).

        Args:
            path: File path to load.

        Returns:
            Loaded (or simulated) DataFrame.
        """
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix in (".csv", ".tsv"):
                return pd.read_csv(path)
            else:
                logger.warning("Unknown file extension: %s. Trying CSV parse.", path.suffix)
                return pd.read_csv(path)

        logger.warning("Data file not found at %s — generating synthetic data.", path)
        return self._generate_synthetic_data(n=1000, drift=("prod" in str(path).lower()))

    @staticmethod
    def _generate_synthetic_data(n: int = 1000, drift: bool = False) -> pd.DataFrame:
        """Generate synthetic RFM + behavioural features for testing.

        Args:
            n: Number of rows.
            drift: If True, add distributional drift to simulate production shift.

        Returns:
            Synthetic DataFrame with model features and label/prediction columns.
        """
        rng = np.random.default_rng(99 if not drift else 42)
        df = pd.DataFrame({
            "recency_days": rng.uniform(1, 90, n) + (30 if drift else 0),
            "frequency": rng.integers(1, 40, n).astype(float),
            "monetary": rng.uniform(20, 3000, n),
            "avg_basket_size": rng.uniform(10, 150, n),
            "rfm_score": rng.uniform(1, 5, n),
            "rolling_mean_7d": rng.uniform(10, 200, n) * (1.15 if drift else 1.0),
            "lag_1d": rng.uniform(0, 250, n),
            "day_of_week": rng.integers(0, 6, n).astype(float),
            "is_weekend": rng.integers(0, 1, n).astype(float),
            "temp_c": rng.uniform(-5, 35, n),
            "cpi_index": rng.uniform(95, 115, n) + (3 if drift else 0),
            "days_to_next_holiday": rng.integers(1, 90, n).astype(float),
            "target": rng.integers(0, 1, n),
            "prediction": rng.uniform(0, 1, n),
        })
        if drift:
            # Simulate drift in key features
            df.loc[:200, "recency_days"] = rng.uniform(60, 180, 201)
            df.loc[:100, "frequency"] = rng.integers(1, 5, 101)
        return df

    # ------------------------------------------------------------------
    # Data drift
    # ------------------------------------------------------------------

    def compute_data_drift(
        self,
        features: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compute PSI and KS-statistic per feature.

        Uses Evidently's DataDriftPreset when available; otherwise falls
        back to scipy-based PSI and KS calculations.

        Args:
            features: List of feature column names to assess. If None,
                uses all shared numeric columns between reference and production.

        Returns:
            Dict of ``{feature: {psi, ks_stat, drift_detected, p_value}}``.
        """
        if features is None:
            numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c in self.production_data.columns]

        results: dict[str, dict[str, Any]] = {}

        if _EVIDENTLY_AVAILABLE:
            try:
                report = Report(metrics=[DataDriftPreset()])
                report.run(
                    reference_data=self.reference_data[features],
                    current_data=self.production_data[features],
                )
                report_dict = report.as_dict()
                metrics = report_dict.get("metrics", [{}])[0].get("result", {})
                drift_by_col = metrics.get("drift_by_columns", {})
                for feat in features:
                    col_result = drift_by_col.get(feat, {})
                    results[feat] = {
                        "psi": float(col_result.get("stattest_threshold", 0.1)),
                        "ks_stat": float(col_result.get("drift_score", 0.0)),
                        "drift_detected": bool(col_result.get("drift_detected", False)),
                        "p_value": float(col_result.get("p_value", 1.0)),
                    }
                logger.info("Evidently drift computed for %d features.", len(results))
                return results
            except Exception as exc:
                logger.warning("Evidently drift computation failed: %s. Using fallback.", exc)

        # Fallback: scipy KS + manual PSI
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            logger.warning("scipy not installed; KS stat set to 0.")
            ks_2samp = None

        for feat in features:
            ref_vals = self.reference_data[feat].dropna().to_numpy()
            prod_vals = self.production_data[feat].dropna().to_numpy()
            if len(ref_vals) == 0 or len(prod_vals) == 0:
                results[feat] = {"psi": 0.0, "ks_stat": 0.0, "drift_detected": False, "p_value": 1.0}
                continue

            psi = self._compute_psi(ref_vals, prod_vals)
            ks_stat, p_value = (0.0, 1.0)
            if ks_2samp is not None:
                ks_result = ks_2samp(ref_vals, prod_vals)
                ks_stat = float(ks_result.statistic)
                p_value = float(ks_result.pvalue)

            results[feat] = {
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "drift_detected": psi > 0.20 or p_value < 0.05,
                "p_value": round(p_value, 4),
            }
        return results

    @staticmethod
    def _compute_psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI) between two distributions.

        PSI = Σ (prod_pct - ref_pct) * ln(prod_pct / ref_pct)

        Interpretation:
        - PSI < 0.10 → No significant drift.
        - 0.10 ≤ PSI < 0.20 → Moderate drift — investigate.
        - PSI ≥ 0.20 → Severe drift — consider retraining.

        Args:
            reference: Reference distribution array (training data).
            production: Production distribution array (current scoring data).
            n_bins: Number of equal-width bins for discretisation.

        Returns:
            PSI value (non-negative float).
        """
        eps = 1e-8
        min_val = min(reference.min(), production.min())
        max_val = max(reference.max(), production.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        bins[-1] += eps  # Include max in last bin

        ref_counts, _ = np.histogram(reference, bins=bins)
        prod_counts, _ = np.histogram(production, bins=bins)

        ref_pct = (ref_counts / len(reference)).clip(min=eps)
        prod_pct = (prod_counts / len(production)).clip(min=eps)

        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return max(0.0, psi)

    # ------------------------------------------------------------------
    # Model performance drift
    # ------------------------------------------------------------------

    def compute_model_performance(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        task: str = "classification",
    ) -> dict[str, Any]:
        """Compute model performance drift metrics using Evidently.

        Compares current-period performance against reference-period performance
        stored in the reference dataset's ``target`` and ``prediction`` columns.

        Args:
            y_true: Ground-truth labels (current period).
            y_pred: Model predictions (current period — probabilities for
                classification, values for regression).
            task: "classification" or "regression".

        Returns:
            Dict with metric values. For classification: auc_roc, f1, log_loss.
            For regression: rmse, mape, r_squared. Includes drift_detected flag.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        result: dict[str, Any] = {"task": task, "n_samples": len(y_true)}

        if task == "classification":
            try:
                from sklearn.metrics import f1_score, log_loss, roc_auc_score
                result["auc_roc"] = round(float(roc_auc_score(y_true, y_pred)), 4)
                y_pred_bin = (y_pred >= 0.5).astype(int)
                result["f1"] = round(float(f1_score(y_true, y_pred_bin, zero_division=0)), 4)
                result["log_loss"] = round(float(log_loss(y_true, y_pred)), 4)
                result["drift_detected"] = result["auc_roc"] < 0.85
            except Exception as exc:
                logger.warning("Classification metrics failed: %s", exc)
                result["drift_detected"] = False

        elif task == "regression":
            try:
                from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
                    result["mape"] = round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 4)
                    result["r_squared"] = round(float(r2_score(y_true, y_pred)), 4)
                result["drift_detected"] = result["mape"] > 12.0  # >10% target + 20% tolerance
            except Exception as exc:
                logger.warning("Regression metrics failed: %s", exc)
                result["drift_detected"] = False

        logger.info("Model performance: %s", result)
        return result

    # ------------------------------------------------------------------
    # Prediction drift
    # ------------------------------------------------------------------

    def compute_prediction_drift(
        self,
        predictions_ref: np.ndarray | pd.Series,
        predictions_prod: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Compute Jensen-Shannon divergence on prediction output distributions.

        JS divergence is a symmetric measure in [0, 1]:
        - 0 → identical distributions.
        - 1 → maximally different distributions.

        A JS divergence > 0.1 warrants investigation.

        Args:
            predictions_ref: Reference-period model predictions.
            predictions_prod: Production-period model predictions.

        Returns:
            Dict with ``js_divergence``, ``drift_detected``, and
            ``psi`` (computed on the prediction column).
        """
        ref_arr = np.asarray(predictions_ref).flatten()
        prod_arr = np.asarray(predictions_prod).flatten()

        try:
            from scipy.spatial.distance import jensenshannon
            js_div = float(jensenshannon(
                np.histogram(ref_arr, bins=20, density=True)[0] + 1e-8,
                np.histogram(prod_arr, bins=20, density=True)[0] + 1e-8,
            ))
        except ImportError:
            # Manual JS divergence
            bins = np.linspace(min(ref_arr.min(), prod_arr.min()), max(ref_arr.max(), prod_arr.max()), 21)
            p, _ = np.histogram(ref_arr, bins=bins, density=True)
            q, _ = np.histogram(prod_arr, bins=bins, density=True)
            p = p + 1e-8
            q = q + 1e-8
            m = 0.5 * (p + q)
            js_div = float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))
            js_div = min(js_div, 1.0)

        psi = self._compute_psi(ref_arr, prod_arr)

        return {
            "js_divergence": round(js_div, 4),
            "psi": round(psi, 4),
            "drift_detected": js_div > 0.10 or psi > 0.20,
        }

    # ------------------------------------------------------------------
    # HTML report generation
    # ------------------------------------------------------------------

    def generate_html_report(self, output_path: str | Path) -> str:
        """Generate a full Evidently HTML monitoring report.

        Includes DataDriftPreset, DataQualityPreset, and ClassificationPreset.
        Saves the report locally and returns the path.

        Args:
            output_path: Local file path to save the HTML report.

        Returns:
            String path of the saved HTML report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if _EVIDENTLY_AVAILABLE:
            try:
                report = Report(
                    metrics=[
                        DataDriftPreset(),
                        DataQualityPreset(),
                        ClassificationPreset(),
                    ]
                )
                column_mapping = ColumnMapping(
                    target="target",
                    prediction="prediction",
                )
                report.run(
                    reference_data=self.reference_data,
                    current_data=self.production_data,
                    column_mapping=column_mapping,
                )
                report.save_html(str(output_path))
                logger.info("Evidently HTML report saved to %s", output_path)
                return str(output_path)
            except Exception as exc:
                logger.error("Failed to generate Evidently HTML report: %s", exc)

        # Fallback: write a minimal HTML summary
        drift_results = self.compute_data_drift()
        html_lines = [
            "<html><head><title>NeuralRetail Drift Report</title></head>",
            "<body><h1>NeuralRetail Drift Monitor</h1>",
            f"<p>Generated: {datetime.utcnow().isoformat()} UTC</p>",
            f"<p>Model: {self.model_name}</p>",
            "<table border='1'><tr><th>Feature</th><th>PSI</th><th>Drift Detected</th></tr>",
        ]
        for feat, info in drift_results.items():
            html_lines.append(
                f"<tr><td>{feat}</td><td>{info['psi']:.4f}</td>"
                f"<td>{'YES' if info['drift_detected'] else 'No'}</td></tr>"
            )
        html_lines += ["</table></body></html>"]
        output_path.write_text("\n".join(html_lines), encoding="utf-8")
        logger.info("Fallback drift report saved to %s", output_path)
        return str(output_path)

    # ------------------------------------------------------------------
    # PSI summary extraction
    # ------------------------------------------------------------------

    def extract_psi_summary(self, report_path: str | Path) -> dict[str, float]:
        """Extract PSI scores from a saved Evidently report or cached drift results.

        In production, Evidently saves a JSON metrics summary alongside the HTML.
        This method attempts to load it. If not found, re-runs the drift computation.

        Args:
            report_path: Path to the HTML report (looks for adjacent .json file).

        Returns:
            Dict of ``{feature: psi}`` for all monitored features, plus
            ``_overall_drift_score`` (mean PSI across all features).
        """
        report_path = Path(report_path)
        json_path = report_path.with_suffix(".json")

        if json_path.exists():
            try:
                raw = json.loads(json_path.read_text())
                psi_data: dict[str, float] = {}
                for feat, val in raw.items():
                    if feat.startswith("_"):
                        continue
                    psi_data[feat] = float(val) if isinstance(val, (int, float)) else 0.0
                overall = float(np.mean(list(psi_data.values()))) if psi_data else 0.0
                psi_data["_overall_drift_score"] = round(overall, 4)
                return psi_data
            except Exception as exc:
                logger.warning("Could not parse PSI JSON: %s", exc)

        # Re-compute PSI
        drift_results = self.compute_data_drift()
        psi_summary: dict[str, float] = {feat: info["psi"] for feat, info in drift_results.items()}
        psi_summary["_overall_drift_score"] = round(float(np.mean(list(psi_summary.values()))), 4)

        # Cache to JSON
        try:
            json_path.write_text(json.dumps(psi_summary, indent=2), encoding="utf-8")
        except Exception:
            pass

        return psi_summary

    # ------------------------------------------------------------------
    # Retrain decision
    # ------------------------------------------------------------------

    def should_trigger_retrain(
        self,
        psi_summary: dict[str, float],
        mape_current: float,
        mape_baseline: float,
        psi_threshold: float = 0.20,
        mape_degradation_threshold: float = 0.15,
    ) -> bool:
        """Decide whether to trigger model retraining.

        Retrain is triggered if:
        - Any feature PSI > ``psi_threshold`` (default 0.20), OR
        - Current MAPE > baseline MAPE * (1 + ``mape_degradation_threshold``).

        This implements the champion/challenger gate from the retraining SLA.

        Args:
            psi_summary: Dict of ``{feature: psi}`` from :meth:`extract_psi_summary`.
            mape_current: Current period demand MAPE (%).
            mape_baseline: Baseline (deployed model) demand MAPE (%).
            psi_threshold: PSI level triggering retrain (default 0.20).
            mape_degradation_threshold: Allowed MAPE degradation fraction (default 15%).

        Returns:
            True if retraining should be triggered; False otherwise.
        """
        # Check PSI threshold
        feature_psis = {k: v for k, v in psi_summary.items() if not k.startswith("_")}
        max_psi = max(feature_psis.values(), default=0.0)
        max_psi_feat = max(feature_psis, key=feature_psis.get, default="N/A")
        psi_trigger = max_psi > psi_threshold

        # Check MAPE degradation
        mape_threshold = mape_baseline * (1 + mape_degradation_threshold)
        mape_trigger = mape_current > mape_threshold

        trigger = psi_trigger or mape_trigger

        logger.info(
            "Retrain decision: psi_trigger=%s (max_psi=%.3f on %s), "
            "mape_trigger=%s (current=%.2f%% baseline=%.2f%% threshold=%.2f%%) → %s",
            psi_trigger, max_psi, max_psi_feat,
            mape_trigger, mape_current, mape_baseline, mape_threshold,
            "TRIGGER" if trigger else "OK",
        )
        return trigger
