"""OpenLineage lineage emitter for NeuralRetail ingestion pipeline.

Emits start and completion events to a Marquez-compatible lineage server,
capturing job metadata, input/output datasets, schema facets, and data quality
scores to enable full pipeline observability.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import (
    Dataset,
    Job,
    Run,
    RunEvent,
    RunState,
)
from openlineage.client.facet_v2 import (
    documentation_job,
    nominal_time_run,
    output_statistics_output_dataset,
    schema_dataset,
)

from neuralretail.src.ingestion.config import MARQUEZ_URL

logger = logging.getLogger(__name__)

NAMESPACE = "neuralretail"


def _build_client() -> OpenLineageClient:
    """Instantiate the OpenLineage HTTP client.

    Returns:
        Configured OpenLineageClient targeting the Marquez URL.
    """
    return OpenLineageClient(url=MARQUEZ_URL)


def _build_schema_facet(fields: list[dict[str, str]]) -> schema_dataset.SchemaDatasetFacet:
    """Build an OpenLineage schema facet from a list of field definitions.

    Args:
        fields: List of dicts with keys ``name`` and ``type``.

    Returns:
        SchemaDatasetFacet with the provided field definitions.
    """
    schema_fields = [
        schema_dataset.SchemaDatasetFacetFields(name=f["name"], type=f["type"])
        for f in fields
    ]
    return schema_dataset.SchemaDatasetFacet(fields=schema_fields)


def emit_start(
    job_name: str,
    run_id: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> None:
    """Emit a START lineage event for an ingestion job.

    Args:
        job_name: Name of the Airflow/Spark job (e.g., "ingest_pos").
        run_id: Unique UUID string for this run.
        inputs: List of input dataset descriptors, each with keys:
            ``name`` (str), ``namespace`` (str), ``fields`` (list[dict]).
        outputs: List of output dataset descriptors with the same keys.

    Example:
        >>> emit_start(
        ...     "ingest_pos",
        ...     str(uuid.uuid4()),
        ...     [{"name": "raw/pos", "namespace": "s3", "fields": []}],
        ...     [{"name": "bronze/pos_transactions", "namespace": "delta", "fields": []}],
        ... )
    """
    client = _build_client()

    input_datasets = [
        Dataset(
            namespace=inp.get("namespace", NAMESPACE),
            name=inp["name"],
            facets={"schema": _build_schema_facet(inp.get("fields", []))},
        )
        for inp in inputs
    ]

    output_datasets = [
        Dataset(
            namespace=out.get("namespace", NAMESPACE),
            name=out["name"],
            facets={"schema": _build_schema_facet(out.get("fields", []))},
        )
        for out in outputs
    ]

    event = RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now(tz=timezone.utc).isoformat(),
        run=Run(
            runId=run_id,
            facets={
                "nominalTime": nominal_time_run.NominalTimeRunFacet(
                    nominalStartTime=datetime.now(tz=timezone.utc).isoformat()
                )
            },
        ),
        job=Job(
            namespace=NAMESPACE,
            name=job_name,
            facets={
                "documentation": documentation_job.DocumentationJobFacet(
                    description=f"NeuralRetail bronze ingestion job: {job_name}"
                )
            },
        ),
        inputs=input_datasets,
        outputs=output_datasets,
        producer="neuralretail-ingestion/v1",
    )

    try:
        client.emit(event)
        logger.info("Lineage START event emitted for job=%s run=%s", job_name, run_id)
    except Exception as exc:
        logger.warning("Lineage emission failed (non-fatal): %s", exc)


def emit_complete(
    run_id: str,
    job_name: str,
    row_count: int,
    dq_score: float,
    output_name: str = "bronze_output",
) -> None:
    """Emit a COMPLETE lineage event with output statistics.

    Args:
        run_id: UUID string matching the START event run_id.
        job_name: Name of the job, must match the name used in emit_start.
        row_count: Number of rows written to the output dataset.
        dq_score: Data quality score (0.0–1.0) from Great Expectations.
        output_name: Name of the primary output dataset (default: "bronze_output").
    """
    client = _build_client()

    output_dataset = Dataset(
        namespace=NAMESPACE,
        name=output_name,
        facets={
            "outputStatistics": output_statistics_output_dataset.OutputStatisticsOutputDatasetFacet(
                rowCount=row_count,
                size=None,
            )
        },
        outputFacets={},
    )

    event = RunEvent(
        eventType=RunState.COMPLETE,
        eventTime=datetime.now(tz=timezone.utc).isoformat(),
        run=Run(
            runId=run_id,
            facets={
                "nominalTime": nominal_time_run.NominalTimeRunFacet(
                    nominalStartTime=datetime.now(tz=timezone.utc).isoformat(),
                    nominalEndTime=datetime.now(tz=timezone.utc).isoformat(),
                )
            },
        ),
        job=Job(namespace=NAMESPACE, name=job_name),
        inputs=[],
        outputs=[output_dataset],
        producer="neuralretail-ingestion/v1",
    )

    try:
        client.emit(event)
        logger.info(
            "Lineage COMPLETE event emitted for run=%s rows=%d dq_score=%.4f",
            run_id,
            row_count,
            dq_score,
        )
    except Exception as exc:
        logger.warning("Lineage COMPLETE emission failed (non-fatal): %s", exc)


def generate_run_id() -> str:
    """Generate a new UUID4 run identifier.

    Returns:
        UUID4 string suitable for use as an OpenLineage run ID.
    """
    return str(uuid.uuid4())
