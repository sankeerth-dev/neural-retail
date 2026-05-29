"""Airflow DAG — NeuralRetail Data Quality Checkpoint.

Triggered by the bronze ingestion DAG via TriggerDagRunOperator.
Runs Great Expectations checkpoints for POS and ERP bronze tables,
evaluates the overall DQ score, pushes metrics to Prometheus, and
sends a Slack alert on failure.

DAG ID: neuralretail_dq_checkpoint
Schedule: None (triggered externally)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "neuralretail-ds",
    "depends_on_past": False,
    "email": ["ds-alerts@amdox.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=1),
}

DQ_THRESHOLD = 0.98


def _slack_alert(context: dict[str, Any]) -> None:
    """Send a Slack alert on task failure.

    Args:
        context: Airflow context dictionary.
    """
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    logger.error("SLACK ALERT: DQ DAG=%s Task=%s FAILED", dag_id, task_id)


def _run_ge_checkpoint_pos(**kwargs: Any) -> None:
    """Execute Great Expectations checkpoint for POS bronze table.

    Raises:
        ValueError: If DQ score is below the 98% threshold.
    """
    from neuralretail.configs.ge_suite_bronze import build_pos_suite, evaluate_suite, DQThresholdError

    suite = build_pos_suite()
    try:
        score = evaluate_suite(suite, table="pos")
        kwargs["ti"].xcom_push(key="pos_dq_score", value=score)
        logger.info("POS DQ checkpoint PASSED: score=%.4f", score)
    except DQThresholdError as exc:
        raise ValueError(str(exc)) from exc


def _run_ge_checkpoint_inventory(**kwargs: Any) -> None:
    """Execute Great Expectations checkpoint for ERP inventory bronze table.

    Raises:
        ValueError: If DQ score is below the 98% threshold.
    """
    from neuralretail.configs.ge_suite_bronze import build_erp_suite, evaluate_suite, DQThresholdError

    suite = build_erp_suite()
    try:
        score = evaluate_suite(suite, table="erp")
        kwargs["ti"].xcom_push(key="erp_dq_score", value=score)
        logger.info("ERP DQ checkpoint PASSED: score=%.4f", score)
    except DQThresholdError as exc:
        raise ValueError(str(exc)) from exc


def _evaluate_overall_dq_score(**kwargs: Any) -> None:
    """Compute overall DQ score and fail the DAG if below threshold.

    Raises:
        ValueError: If either POS or ERP score is below DQ_THRESHOLD.
    """
    ti = kwargs["ti"]
    pos_score = ti.xcom_pull(key="pos_dq_score", task_ids="run_ge_checkpoint_pos") or 0.0
    erp_score = ti.xcom_pull(key="erp_dq_score", task_ids="run_ge_checkpoint_inventory") or 0.0
    overall_score = (pos_score + erp_score) / 2.0

    logger.info(
        "DQ scores — POS=%.4f ERP=%.4f Overall=%.4f threshold=%.4f",
        pos_score,
        erp_score,
        overall_score,
        DQ_THRESHOLD,
    )

    if overall_score < DQ_THRESHOLD:
        raise ValueError(
            f"Overall DQ score {overall_score:.4f} < threshold {DQ_THRESHOLD}. "
            "Halting pipeline to prevent low-quality features entering silver layer."
        )

    kwargs["ti"].xcom_push(key="overall_dq_score", value=overall_score)


def _push_dq_metrics_to_prometheus(**kwargs: Any) -> None:
    """Push DQ metrics to Prometheus pushgateway.

    Falls back to logging if the Prometheus client or pushgateway is unavailable.
    """
    ti = kwargs["ti"]
    pos_score = ti.xcom_pull(key="pos_dq_score", task_ids="run_ge_checkpoint_pos") or 0.0
    erp_score = ti.xcom_pull(key="erp_dq_score", task_ids="run_ge_checkpoint_inventory") or 0.0

    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        g_pos = Gauge("nr_dq_score_pos", "POS DQ score", registry=registry)
        g_erp = Gauge("nr_dq_score_erp", "ERP DQ score", registry=registry)
        g_pos.set(pos_score)
        g_erp.set(erp_score)
        push_to_gateway("localhost:9091", job="neuralretail_dq", registry=registry)
        logger.info("DQ metrics pushed to Prometheus gateway")
    except Exception as exc:
        logger.warning(
            "Prometheus push failed (non-fatal): %s. Scores — POS=%.4f ERP=%.4f",
            exc,
            pos_score,
            erp_score,
        )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="neuralretail_dq_checkpoint",
    default_args=DEFAULT_ARGS,
    description="Data quality checkpoint triggered after bronze ingestion",
    schedule=None,  # Triggered via TriggerDagRunOperator
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["neuralretail", "data-quality"],
    on_failure_callback=_slack_alert,
    doc_md=__doc__,
) as dag:

    t_pos_checkpoint = PythonOperator(
        task_id="run_ge_checkpoint_pos",
        python_callable=_run_ge_checkpoint_pos,
        on_failure_callback=_slack_alert,
    )

    t_erp_checkpoint = PythonOperator(
        task_id="run_ge_checkpoint_inventory",
        python_callable=_run_ge_checkpoint_inventory,
        on_failure_callback=_slack_alert,
    )

    t_evaluate_overall = PythonOperator(
        task_id="evaluate_overall_dq_score",
        python_callable=_evaluate_overall_dq_score,
        on_failure_callback=_slack_alert,
    )

    t_push_metrics = PythonOperator(
        task_id="push_dq_metrics_to_prometheus",
        python_callable=_push_dq_metrics_to_prometheus,
    )

    t_pos_checkpoint >> t_erp_checkpoint >> t_evaluate_overall >> t_push_metrics
