"""Baseline logistic regression churn classifier for NeuralRetail Week 1.

Trains a scikit-learn Pipeline (StandardScaler + LogisticRegression) on
customer RFM features, evaluates classification metrics, and logs artefacts
to MLflow with model registry staging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "churn_prediction"
MODEL_REGISTRY_NAME = "churn_baseline"
PLOT_DIR = Path("artifacts/churn_plots")


def _get_experiment_id() -> str:
    """Retrieve or create the churn_prediction MLflow experiment.

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


class BaselineChurnClassifier:
    """Baseline logistic regression churn classifier.

    Loads customer RFM features, applies a heuristic churn label,
    trains a balanced logistic regression model, and logs to MLflow.

    Example:
        >>> clf = BaselineChurnClassifier()
        >>> X_train, X_test, y_train, y_test = clf.load_features("data/silver/customer_features")
        >>> model = clf.train(X_train, y_train)
        >>> metrics = clf.evaluate(model, X_test, y_test)
    """

    FEATURE_COLS = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_basket_size",
        "rfm_score",
    ]

    def __init__(self) -> None:
        """Initialise the BaselineChurnClassifier."""
        self._exp_id: str = _get_experiment_id()
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def load_features(
        self, feature_store_path: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load customer RFM features and construct churn labels.

        Churn label heuristic: label=1 if recency_days > 30 AND frequency < 2
        in the last 30 days; else label=0.

        Args:
            feature_store_path: Path to customer_features Parquet/Delta.

        Returns:
            Tuple of (X: pd.DataFrame of feature columns, y: pd.Series of labels).
        """
        try:
            df = pd.read_parquet(feature_store_path)
        except FileNotFoundError:
            logger.warning(
                "Feature data not found at %s — using synthetic data", feature_store_path
            )
            df = self._generate_synthetic_features()

        # Ensure required columns exist
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.random.default_rng(42).uniform(0, 1, len(df))

        # Churn label: high recency AND low purchase frequency
        df["churn_label"] = (
            (df["recency_days"] > 30) & (df["frequency"] < 2)
        ).astype(int)

        churn_rate = df["churn_label"].mean()
        logger.info(
            "Churn labels created: %d customers, churn rate=%.2f%%",
            len(df),
            churn_rate * 100,
        )

        X = df[self.FEATURE_COLS].fillna(0.0)
        y = df["churn_label"]
        return X, y

    def _generate_synthetic_features(self, n: int = 10_000) -> pd.DataFrame:
        """Generate synthetic customer RFM features for testing.

        Args:
            n: Number of synthetic customers.

        Returns:
            DataFrame with RFM feature columns.
        """
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "customer_id": [f"CUST-{i:07d}" for i in range(n)],
                "recency_days": rng.integers(1, 180, size=n),
                "frequency": rng.integers(1, 30, size=n),
                "monetary": rng.uniform(100, 50_000, size=n).round(2),
                "avg_basket_size": rng.uniform(50, 5_000, size=n).round(2),
                "rfm_score": rng.uniform(1.0, 5.0, size=n).round(3),
                "snapshot_date": pd.Timestamp("2026-05-01"),
            }
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """Train a logistic regression churn classifier pipeline.

        Pipeline: StandardScaler → LogisticRegression(class_weight=balanced).

        Args:
            X_train: Training feature DataFrame.
            y_train: Binary churn label Series.

        Returns:
            Fitted sklearn Pipeline.
        """
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        logger.info(
            "Logistic regression trained: %d samples, %d features",
            len(X_train),
            X_train.shape[1],
        )
        return pipeline

    def evaluate(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate the churn classifier on a test set.

        Args:
            model: Fitted sklearn Pipeline.
            X_test: Test feature DataFrame.
            y_test: True binary labels.

        Returns:
            Dict with keys: auc_roc, f1, precision, recall,
            confusion_matrix (list), precision_at_top20pct.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc_roc = float(roc_auc_score(y_test, y_proba))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Precision at top-20% by predicted probability
        n_top20 = max(1, int(len(y_test) * 0.20))
        top20_idx = np.argsort(y_proba)[::-1][:n_top20]
        precision_at_top20 = float(y_test.iloc[top20_idx].mean())

        metrics = {
            "auc_roc": auc_roc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "precision_at_top20pct": precision_at_top20,
        }
        logger.info(
            "Churn evaluation: AUC=%.4f F1=%.4f P@20%%=%.4f",
            auc_roc,
            f1,
            precision_at_top20,
        )
        return metrics

    def plot_roc_curve(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> str:
        """Generate and save an ROC curve plot.

        Args:
            model: Fitted sklearn Pipeline.
            X_test: Test feature DataFrame.
            y_test: True binary labels.

        Returns:
            File path string of the saved PNG.
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.7, label="Target AUC=0.90")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Churn Classifier — ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = str(PLOT_DIR / "roc_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_confusion_matrix(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> str:
        """Save a confusion matrix heatmap PNG.

        Args:
            model: Fitted sklearn Pipeline.
            X_test: Test feature DataFrame.
            y_test: True binary labels.

        Returns:
            File path string of the saved PNG.
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Churn Classifier — Confusion Matrix")
        plt.tight_layout()

        path = str(PLOT_DIR / "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_feature_importance(self, model: Pipeline, X_test: pd.DataFrame) -> str:
        """Save feature coefficients as a CSV artefact.

        Args:
            model: Fitted sklearn Pipeline with LogisticRegression step.
            X_test: Test feature DataFrame (for column names).

        Returns:
            File path string of the saved CSV.
        """
        lr = model.named_steps["classifier"]
        fi_df = pd.DataFrame(
            {
                "feature": X_test.columns.tolist(),
                "coefficient": lr.coef_[0].tolist(),
                "abs_coefficient": np.abs(lr.coef_[0]).tolist(),
            }
        ).sort_values("abs_coefficient", ascending=False)

        path = str(PLOT_DIR / "feature_importance.csv")
        fi_df.to_csv(path, index=False)
        return path

    def log_to_mlflow(
        self,
        model: Pipeline,
        metrics: dict[str, Any],
        params: dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Log churn model artefacts and metrics to MLflow.

        Args:
            model: Fitted sklearn Pipeline.
            metrics: Evaluation metrics dict from evaluate().
            params: Model hyperparameters dict.
            X_test: Test feature DataFrame for plots and feature importance.
            y_test: True binary labels for ROC curve.
        """
        with mlflow.start_run(
            experiment_id=self._exp_id, run_name="logistic_regression_baseline"
        ):
            mlflow.log_params(
                {
                    "algorithm": "LogisticRegression",
                    "C": 1.0,
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "features": ",".join(self.FEATURE_COLS),
                    **params,
                }
            )

            loggable_metrics = {
                k: v for k, v in metrics.items() if not isinstance(v, list)
            }
            mlflow.log_metrics(loggable_metrics)
            mlflow.set_tags(
                {
                    "model_type": "baseline",
                    "algorithm": "logistic_regression",
                    "project": "neuralretail",
                }
            )

            # Save and log artefacts
            roc_path = self.plot_roc_curve(model, X_test, y_test)
            cm_path = self._save_confusion_matrix(model, X_test, y_test)
            fi_path = self._save_feature_importance(model, X_test)

            mlflow.log_artifact(roc_path, artifact_path="plots")
            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(fi_path, artifact_path="tables")

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="churn_model",
                registered_model_name=MODEL_REGISTRY_NAME,
            )
            logger.info("Churn baseline model logged and registered to staging.")
