"""Silver layer writer for NeuralRetail feature tables.

Writes customer RFM features and SKU demand features to Delta Lake silver
tables using MERGE (upsert) semantics to ensure idempotency. Logs
write statistics to MLflow.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import polars as pl
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from neuralretail.src.ingestion.config import (
    DELTA_PACKAGE,
    MLFLOW_TRACKING_URI,
    SILVER_CUSTOMER_FEATURES,
    SILVER_SKU_DEMAND_FEATURES,
    SPARK_APP_NAME,
)

logger = logging.getLogger(__name__)


def _get_or_create_spark() -> SparkSession:
    """Return an active SparkSession, creating one if needed.

    Returns:
        Configured SparkSession with Delta Lake support.
    """
    return (
        SparkSession.builder.appName(f"{SPARK_APP_NAME}-SilverWriter")
        .config("spark.jars.packages", DELTA_PACKAGE)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


class SilverWriter:
    """Writes feature DataFrames to Delta Lake silver tables using MERGE semantics.

    Ensures idempotent writes via Delta MERGE (upsert) on natural keys.
    Logs row counts and feature counts to MLflow for observability.

    Example:
        >>> writer = SilverWriter()
        >>> writer.write_customer_features(rfm_df)
        >>> writer.write_sku_demand_features(demand_df)
    """

    def __init__(self) -> None:
        """Initialise SilverWriter with a SparkSession and MLflow tracking URI."""
        self._spark: SparkSession = _get_or_create_spark()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def write_customer_features(self, df: pl.DataFrame) -> None:
        """Write customer RFM features to the silver customer_features Delta table.

        Uses Delta MERGE on (customer_id, snapshot_date) to support idempotent
        daily refreshes. Partitioned by snapshot_date.

        Args:
            df: Polars DataFrame with customer feature columns including
                customer_id and snapshot_date.
        """
        spark_df = self._spark.createDataFrame(df.to_pandas())
        target_path = SILVER_CUSTOMER_FEATURES

        try:
            delta_table = DeltaTable.forPath(self._spark, target_path)
            (
                delta_table.alias("target")
                .merge(
                    spark_df.alias("source"),
                    "target.customer_id = source.customer_id "
                    "AND target.snapshot_date = source.snapshot_date",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info("Customer features MERGED into %s", target_path)
        except Exception:
            # Table doesn't exist yet — initial write
            (
                spark_df.write.format("delta")
                .mode("overwrite")
                .partitionBy("snapshot_date")
                .save(target_path)
            )
            logger.info("Customer features CREATED at %s", target_path)

        row_count = len(df)
        feature_cols = [c for c in df.columns if c not in ("customer_id", "snapshot_date")]
        self.log_write_stats("silver_customer_features", row_count, feature_cols)

    def write_sku_demand_features(self, df: pl.DataFrame) -> None:
        """Write SKU demand features to the silver sku_demand_features Delta table.

        Uses Delta MERGE on (product_id, date) to support idempotent writes.
        Partitioned by product_id for efficient per-SKU reads.

        Args:
            df: Polars DataFrame with SKU demand feature columns including
                product_id and date.
        """
        spark_df = self._spark.createDataFrame(df.to_pandas())
        target_path = SILVER_SKU_DEMAND_FEATURES

        try:
            delta_table = DeltaTable.forPath(self._spark, target_path)
            (
                delta_table.alias("target")
                .merge(
                    spark_df.alias("source"),
                    "target.product_id = source.product_id "
                    "AND target.date = source.date",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            logger.info("SKU demand features MERGED into %s", target_path)
        except Exception:
            (
                spark_df.write.format("delta")
                .mode("overwrite")
                .partitionBy("product_id")
                .save(target_path)
            )
            logger.info("SKU demand features CREATED at %s", target_path)

        row_count = len(df)
        feature_cols = [c for c in df.columns if c not in ("product_id", "date")]
        self.log_write_stats("silver_sku_demand_features", row_count, feature_cols)

    def log_write_stats(
        self,
        table_name: str,
        row_count: int,
        feature_columns: list[str],
    ) -> None:
        """Log silver write statistics to MLflow.

        Args:
            table_name: Name of the target Delta table.
            row_count: Number of rows written or merged.
            feature_columns: List of feature column names (excludes key columns).
        """
        with mlflow.start_run(run_name=f"silver_write_{table_name}", nested=True):
            mlflow.log_metrics({"row_count": row_count})
            mlflow.log_param("table_name", table_name)
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("feature_columns", ",".join(feature_columns[:50]))
            logger.info(
                "MLflow logged: table=%s rows=%d features=%d",
                table_name,
                row_count,
                len(feature_columns),
            )
