"""MLflow experiment setup for NeuralRetail Week 1.

Creates the four core MLflow experiments used across the Week 1 baseline
model runs. Safe to run multiple times (skips existing experiments).
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"

EXPERIMENTS = [
    {
        "name": "demand_forecasting",
        "tags": {
            "team": "ds",
            "project": "neuralretail",
            "objective": "MAPE_leq_10pct_30day",
            "model_types": "prophet,lstm,xgboost,ensemble",
        },
    },
    {
        "name": "churn_prediction",
        "tags": {
            "team": "ds",
            "project": "neuralretail",
            "objective": "AUC_ROC_geq_0.90",
            "model_types": "logistic,xgboost,lightgbm,stacking",
        },
    },
    {
        "name": "customer_segmentation",
        "tags": {
            "team": "ds",
            "project": "neuralretail",
            "objective": "silhouette_geq_0.45",
            "model_types": "kmeans,dbscan,gmm",
        },
    },
    {
        "name": "price_elasticity",
        "tags": {
            "team": "ds",
            "project": "neuralretail",
            "objective": "causal_revenue_lift",
            "model_types": "dowhy,econml,dml",
        },
    },
]


def setup_experiments() -> dict[str, str]:
    """Create MLflow experiments for NeuralRetail Week 1.

    Connects to the MLflow tracking server, creates each experiment if it
    does not already exist, and prints the experiment IDs.

    Returns:
        Dict mapping experiment name to experiment ID string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    created: dict[str, str] = {}

    for exp_config in EXPERIMENTS:
        name = exp_config["name"]
        tags = exp_config["tags"]

        existing = client.get_experiment_by_name(name)
        if existing is not None:
            exp_id = existing.experiment_id
            logger.info("Experiment '%s' already exists: id=%s", name, exp_id)
        else:
            exp_id = mlflow.create_experiment(name=name, tags=tags)
            logger.info("Created experiment '%s': id=%s", name, exp_id)

        created[name] = exp_id
        print(f"  ✓ {name:<30} id={exp_id}")

    return created


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "=" * 50)
    print("  NeuralRetail — MLflow Experiment Setup")
    print("=" * 50)
    result = setup_experiments()
    print("=" * 50)
    print(f"  {len(result)} experiments configured.\n")
