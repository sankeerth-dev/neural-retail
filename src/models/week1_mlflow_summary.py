"""MLflow Week 1 summary report for NeuralRetail.

Queries the demand_forecasting and churn_prediction experiments, builds a
summary DataFrame, prints a tabulate table to stdout, flags out-of-tolerance
rows, and exports to docs/mlflow_week1_summary.csv.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from tabulate import tabulate

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
OUTPUT_CSV = Path("docs/mlflow_week1_summary.csv")

EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    {
        "experiment_name": "demand_forecasting",
        "primary_metric": "mape",
        "target": 0.10,
        "direction": "minimize",
        "registered_model": "prophet_baseline",
    },
    {
        "experiment_name": "churn_prediction",
        "primary_metric": "auc_roc",
        "target": 0.90,
        "direction": "maximize",
        "registered_model": "churn_baseline",
    },
]


def _get_registered_model_stage(client: MlflowClient, model_name: str) -> str:
    """Look up the current stage of the latest registered model version.

    Args:
        client: MLflow tracking client.
        model_name: Registered model name.

    Returns:
        Stage string (e.g., "Staging", "Production") or "Not Registered".
    """
    try:
        versions = client.get_latest_versions(model_name)
        if versions:
            return versions[0].current_stage
        return "Not Registered"
    except Exception:
        return "Not Registered"


def build_week1_summary() -> pd.DataFrame:
    """Query MLflow experiments and build a summary DataFrame.

    Returns:
        Summary DataFrame with columns: run_name, model_type, primary_metric,
        primary_value, target, pass_fail, registered_model, stage.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    records = []

    for config in EXPERIMENT_CONFIGS:
        exp_name = config["experiment_name"]
        primary_metric = config["primary_metric"]
        target = config["target"]
        direction = config["direction"]
        registered_model = config["registered_model"]

        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            logger.warning("Experiment '%s' not found in MLflow", exp_name)
            records.append(
                {
                    "run_name": "N/A",
                    "model_type": exp_name,
                    "primary_metric": primary_metric,
                    "primary_value": None,
                    "target": target,
                    "pass_fail": "NOT RUN",
                    "registered_model": registered_model,
                    "stage": "Not Registered",
                }
            )
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{primary_metric} {'ASC' if direction == 'minimize' else 'DESC'}"],
            max_results=20,
        )

        stage = _get_registered_model_stage(client, registered_model)

        if not runs:
            logger.warning("No runs found for experiment '%s'", exp_name)
            records.append(
                {
                    "run_name": "N/A",
                    "model_type": exp_name,
                    "primary_metric": primary_metric,
                    "primary_value": None,
                    "target": target,
                    "pass_fail": "NOT RUN",
                    "registered_model": registered_model,
                    "stage": stage,
                }
            )
            continue

        for run in runs:
            metric_value = run.data.metrics.get(primary_metric)
            if metric_value is None:
                continue

            if direction == "minimize":
                passes = metric_value <= target
            else:
                passes = metric_value >= target

            model_type = run.data.tags.get("model_type", "unknown")
            algorithm = run.data.tags.get("algorithm", run.data.params.get("algorithm", "unknown"))

            records.append(
                {
                    "run_name": run.info.run_name or run.info.run_id[:8],
                    "model_type": f"{algorithm} ({model_type})",
                    "primary_metric": primary_metric,
                    "primary_value": round(metric_value, 4),
                    "target": target,
                    "pass_fail": "✅ PASS" if passes else "❌ FAIL",
                    "registered_model": registered_model,
                    "stage": stage,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """Generate and print the Week 1 MLflow summary report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 80)
    print("  NeuralRetail — Week 1 MLflow Experiment Summary")
    print("=" * 80)

    try:
        summary_df = build_week1_summary()
    except Exception as exc:
        logger.error("Failed to connect to MLflow at %s: %s", MLFLOW_TRACKING_URI, exc)
        print(f"\n  ⚠️  MLflow not reachable at {MLFLOW_TRACKING_URI}.")
        print("  Run 'docker compose up mlflow' and retry.\n")
        sys.exit(1)

    if summary_df.empty:
        print("\n  No model runs found. Run experiments first.\n")
        sys.exit(0)

    print(
        tabulate(
            summary_df.to_dict("records"),
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
        )
    )

    # Flag out-of-tolerance rows
    failing = summary_df[summary_df["pass_fail"].str.contains("FAIL", na=False)]
    if not failing.empty:
        print(f"\n  ⚠️  {len(failing)} model(s) below target threshold:")
        for _, row in failing.iterrows():
            print(
                f"     {row['registered_model']} {row['primary_metric']}="
                f"{row['primary_value']} (target {'≤' if 'mape' in row['primary_metric'] else '≥'} {row['target']})"
            )
    else:
        print("\n  ✅ All models meet Week 1 targets.\n")

    # Export to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Summary exported to: {OUTPUT_CSV}\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
