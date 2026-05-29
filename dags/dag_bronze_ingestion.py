"""Airflow DAG — NeuralRetail Bronze Ingestion Pipeline.

Scheduled daily at 02:00 UTC. Orchestrates ingestion of POS, e-commerce,
ERP, and external data into Delta Lake bronze tables, followed by Great
Expectations data quality validation and success notification.

DAG ID: neuralretail_bronze_ingestion
Schedule: 0 2 * * *
SLA: 2 hours
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "neuralretail-ds",
    "depends_on_past": False,
    "email": ["ds-alerts@amdox.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def _slack_alert(context: dict) -> None:
    """Send a Slack alert on DAG task failure.

    Args:
        context: Airflow context dictionary passed by on_failure_callback.
    """
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    execution_date = context.get("execution_date")
    logger.error(
        "SLACK ALERT: DAG=%s Task=%s failed at %s",
        dag_id,
        task_id,
        execution_date,
    )
    # In production replace with actual Slack webhook call:
    # SlackWebhookOperator(http_conn_id="slack_conn", message=msg).execute(context)


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------
def _ingest_pos(**kwargs) -> None:
    """Ingest POS transaction data into bronze Delta table."""
    from neuralretail.src.ingestion.spark_ingest import SparkIngestor
    from neuralretail.src.ingestion.lineage import emit_start, emit_complete, generate_run_id
    import os

    source_path = os.environ.get("NR_POS_SOURCE_PATH", "data/raw/pos")
    run_id = generate_run_id()

    emit_start(
        "ingest_pos",
        run_id,
        [{"name": source_path, "namespace": "file"}],
        [{"name": "bronze/pos_transactions", "namespace": "delta"}],
    )

    ingestor = SparkIngestor()
    spark = ingestor.get_spark_session()
    ingestor.ingest_pos(spark, source_path)

    emit_complete(run_id, "ingest_pos", row_count=0, dq_score=1.0, output_name="bronze/pos_transactions")
    kwargs["ti"].xcom_push(key="pos_status", value="SUCCESS")


def _ingest_ecommerce(**kwargs) -> None:
    """Ingest e-commerce event data into bronze Delta table."""
    from neuralretail.src.ingestion.spark_ingest import SparkIngestor
    import os

    source_path = os.environ.get("NR_ECOM_SOURCE_PATH", "data/raw/ecommerce")
    ingestor = SparkIngestor()
    spark = ingestor.get_spark_session()
    ingestor.ingest_ecommerce(spark, source_path)
    kwargs["ti"].xcom_push(key="ecom_status", value="SUCCESS")


def _ingest_erp(**kwargs) -> None:
    """Ingest ERP inventory snapshot data into bronze Delta table."""
    from neuralretail.src.ingestion.spark_ingest import SparkIngestor
    import os

    source_path = os.environ.get("NR_ERP_SOURCE_PATH", "data/raw/erp")
    ingestor = SparkIngestor()
    spark = ingestor.get_spark_session()
    ingestor.ingest_erp(spark, source_path)
    kwargs["ti"].xcom_push(key="erp_status", value="SUCCESS")


def _ingest_external(**kwargs) -> None:
    """Ingest external signals data into bronze Delta table."""
    from neuralretail.src.ingestion.spark_ingest import SparkIngestor
    import os

    source_path = os.environ.get("NR_EXTERNAL_SOURCE_PATH", "data/raw/external")
    ingestor = SparkIngestor()
    spark = ingestor.get_spark_session()
    ingestor.ingest_external(spark, source_path)
    kwargs["ti"].xcom_push(key="external_status", value="SUCCESS")


def _validate_dq_pos(**kwargs) -> None:
    """Run Great Expectations checkpoint for POS bronze data."""
    from neuralretail.configs.ge_suite_bronze import build_pos_suite, evaluate_suite
    from neuralretail.src.ingestion.config import GE_THRESHOLD

    suite = build_pos_suite()
    score = evaluate_suite(suite, table="pos")

    if score < GE_THRESHOLD:
        raise ValueError(
            f"POS DQ score {score:.4f} below threshold {GE_THRESHOLD}. "
            "Investigate data quality issues before proceeding."
        )
    logger.info("POS DQ validation PASSED: score=%.4f", score)
    kwargs["ti"].xcom_push(key="pos_dq_score", value=score)


def _validate_dq_inventory(**kwargs) -> None:
    """Run Great Expectations checkpoint for ERP inventory bronze data."""
    from neuralretail.configs.ge_suite_bronze import build_erp_suite, evaluate_suite
    from neuralretail.src.ingestion.config import GE_THRESHOLD

    suite = build_erp_suite()
    score = evaluate_suite(suite, table="erp")

    if score < GE_THRESHOLD:
        raise ValueError(
            f"ERP DQ score {score:.4f} below threshold {GE_THRESHOLD}."
        )
    logger.info("ERP DQ validation PASSED: score=%.4f", score)
    kwargs["ti"].xcom_push(key="erp_dq_score", value=score)


def _notify_success(**kwargs) -> None:
    """Log success notification and push summary to XCom."""
    ti = kwargs["ti"]
    pos_dq = ti.xcom_pull(key="pos_dq_score", task_ids="validate_dq_pos") or 0.0
    erp_dq = ti.xcom_pull(key="erp_dq_score", task_ids="validate_dq_inventory") or 0.0
    logger.info(
        "Bronze ingestion pipeline SUCCESS. POS DQ=%.4f ERP DQ=%.4f",
        pos_dq,
        erp_dq,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="neuralretail_bronze_ingestion",
    default_args=DEFAULT_ARGS,
    description="Daily bronze layer ingestion: POS, e-commerce, ERP, external",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["neuralretail", "ingestion"],
    on_failure_callback=_slack_alert,
    sla_miss_callback=_slack_alert,
    doc_md=__doc__,
) as dag:

    t_ingest_pos = PythonOperator(
        task_id="ingest_pos",
        python_callable=_ingest_pos,
        sla=timedelta(hours=2),
    )

    t_ingest_ecommerce = PythonOperator(
        task_id="ingest_ecommerce",
        python_callable=_ingest_ecommerce,
        sla=timedelta(hours=2),
    )

    t_ingest_erp = PythonOperator(
        task_id="ingest_erp",
        python_callable=_ingest_erp,
        sla=timedelta(hours=2),
    )

    t_ingest_external = PythonOperator(
        task_id="ingest_external",
        python_callable=_ingest_external,
        sla=timedelta(hours=2),
    )

    t_validate_dq_pos = PythonOperator(
        task_id="validate_dq_pos",
        python_callable=_validate_dq_pos,
        sla=timedelta(hours=2),
    )

    t_validate_dq_inventory = PythonOperator(
        task_id="validate_dq_inventory",
        python_callable=_validate_dq_inventory,
        sla=timedelta(hours=2),
    )

    t_notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=_notify_success,
    )

    # ---------------------------------------------------------------------------
    # Task dependencies
    # ---------------------------------------------------------------------------
    t_ingest_pos >> t_ingest_ecommerce >> t_ingest_erp >> t_ingest_external
    t_ingest_external >> t_validate_dq_pos >> t_validate_dq_inventory >> t_notify_success
