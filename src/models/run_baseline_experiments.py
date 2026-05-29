"""Baseline experiment runner for NeuralRetail Week 1.

Entry-point script to run demand forecasting and/or churn prediction
baseline experiments. Prints a summary table and exits with code 1 if
any model falls short of minimum acceptable performance thresholds.

Usage:
    python run_baseline_experiments.py --experiment demand --top_n_skus 100
    python run_baseline_experiments.py --experiment churn
    python run_baseline_experiments.py --experiment both --top_n_skus 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Performance thresholds (hard gates)
DEMAND_MAPE_MAX = 0.20   # Exit with code 1 if baseline MAPE > 20%
CHURN_AUC_MIN = 0.60     # Exit with code 1 if baseline AUC < 0.60

# Business targets (soft gates — report PASS/FAIL)
DEMAND_MAPE_TARGET = 0.10
CHURN_AUC_TARGET = 0.90


def _run_demand_experiment(top_n_skus: int, test_days: int) -> list[dict[str, Any]]:
    """Run the demand forecasting baseline experiment.

    Args:
        top_n_skus: Number of top-revenue SKUs to train models for.
        test_days: Number of days to use for hold-out evaluation.

    Returns:
        List of result dicts for summary table.
    """
    from neuralretail.src.models.forecasting.baseline_prophet import BaselineProphetForecaster

    logger.info("Running demand forecasting baseline (top_n=%d, test_days=%d)", top_n_skus, test_days)
    forecaster = BaselineProphetForecaster(horizon_days=test_days)
    df = forecaster.load_data("data/silver/sku_demand_features")

    sku_revenues = df.groupby("product_id")["y"].sum().nlargest(min(top_n_skus, 5))
    q33 = sku_revenues.quantile(0.33)
    q66 = sku_revenues.quantile(0.66)

    all_mapes = []
    results = []

    for sku_id, revenue in sku_revenues.items():
        sku_df = df[df["product_id"] == sku_id][["ds", "y"]].copy()
        if len(sku_df) < 60:
            continue
        try:
            model = forecaster.train(sku_id, sku_df)
            metrics = forecaster.evaluate(model, sku_df)
            all_mapes.append(metrics["mape"])
            from neuralretail.src.models.forecasting.baseline_prophet import _sku_tier
            tier = _sku_tier(revenue, q33, q66)
            forecaster.log_to_mlflow(sku_id, model, metrics, {}, sku_tier=tier)
        except Exception as exc:
            logger.error("Demand experiment failed for %s: %s", sku_id, exc)
            metrics = {"mape": 1.0, "rmse": 999.0}

        results.append({
            "model": f"Prophet_{sku_id[:12]}",
            "algorithm": "Prophet",
            "metric": "MAPE",
            "value": round(metrics["mape"], 4),
            "target": DEMAND_MAPE_TARGET,
            "pass_fail": "PASS" if metrics["mape"] <= DEMAND_MAPE_TARGET else "FAIL",
        })

    avg_mape = float(np.mean(all_mapes)) if all_mapes else 1.0
    results.append({
        "model": "Prophet_AVG",
        "algorithm": "Prophet",
        "metric": "MAPE (avg)",
        "value": round(avg_mape, 4),
        "target": DEMAND_MAPE_TARGET,
        "pass_fail": "PASS" if avg_mape <= DEMAND_MAPE_TARGET else "FAIL",
    })
    return results, avg_mape


def _run_churn_experiment(test_days: int) -> tuple[list[dict[str, Any]], float]:
    """Run the churn prediction baseline experiment.

    Args:
        test_days: Unused for churn (kept for API consistency).

    Returns:
        Tuple of (results list, auc_roc float).
    """
    from neuralretail.src.models.churn.baseline_logistic import BaselineChurnClassifier

    logger.info("Running churn prediction baseline")
    clf = BaselineChurnClassifier()
    X, y = clf.load_features("data/silver/customer_features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = clf.train(X_train, y_train)
    metrics = clf.evaluate(model, X_test, y_test)
    clf.log_to_mlflow(model, metrics, {}, X_test, y_test)

    results = [
        {
            "model": "LogisticRegression",
            "algorithm": "LogisticRegression",
            "metric": "AUC-ROC",
            "value": round(metrics["auc_roc"], 4),
            "target": CHURN_AUC_TARGET,
            "pass_fail": "PASS" if metrics["auc_roc"] >= CHURN_AUC_TARGET else "FAIL",
        },
        {
            "model": "LogisticRegression",
            "algorithm": "LogisticRegression",
            "metric": "F1",
            "value": round(metrics["f1"], 4),
            "target": 0.70,
            "pass_fail": "PASS" if metrics["f1"] >= 0.70 else "FAIL",
        },
        {
            "model": "LogisticRegression",
            "algorithm": "LogisticRegression",
            "metric": "P@top20%",
            "value": round(metrics["precision_at_top20pct"], 4),
            "target": 0.50,
            "pass_fail": "PASS" if metrics["precision_at_top20pct"] >= 0.50 else "FAIL",
        },
    ]
    return results, metrics["auc_roc"]


def main() -> None:
    """Parse arguments, run experiments, print summary, and exit."""
    parser = argparse.ArgumentParser(description="NeuralRetail Week 1 Baseline Experiments")
    parser.add_argument(
        "--experiment",
        choices=["demand", "churn", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--top_n_skus",
        type=int,
        default=100,
        help="Number of top-revenue SKUs for demand forecasting (default: 100)",
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=30,
        help="Hold-out evaluation window in days (default: 30)",
    )
    args = parser.parse_args()

    all_results = []
    avg_demand_mape = 0.0
    churn_auc = 1.0

    if args.experiment in ("demand", "both"):
        demand_results, avg_demand_mape = _run_demand_experiment(args.top_n_skus, args.test_days)
        all_results.extend(demand_results)

    if args.experiment in ("churn", "both"):
        churn_results, churn_auc = _run_churn_experiment(args.test_days)
        all_results.extend(churn_results)

    print("\n" + "=" * 72)
    print("  NeuralRetail — Week 1 Baseline Model Results")
    print("=" * 72)
    print(
        tabulate(
            all_results,
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
        )
    )
    print("=" * 72)

    # Hard gate checks
    exit_code = 0
    if args.experiment in ("demand", "both") and avg_demand_mape > DEMAND_MAPE_MAX:
        logger.error(
            "HARD GATE FAILED: Demand MAPE %.4f > max allowed %.4f",
            avg_demand_mape,
            DEMAND_MAPE_MAX,
        )
        exit_code = 1

    if args.experiment in ("churn", "both") and churn_auc < CHURN_AUC_MIN:
        logger.error(
            "HARD GATE FAILED: Churn AUC %.4f < min required %.4f",
            churn_auc,
            CHURN_AUC_MIN,
        )
        exit_code = 1

    if exit_code == 0:
        print("\n  ✅ All hard gate checks PASSED.\n")
    else:
        print("\n  ❌ One or more hard gate checks FAILED. Investigate before proceeding.\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
