"""PySpark ingestion module for NeuralRetail bronze layer.

Handles ingestion of POS transactions, e-commerce events, ERP inventory
snapshots, and external signals into Delta Lake bronze tables with schema
enforcement, metadata enrichment, and partitioned writes.
"""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from neuralretail.src.ingestion.config import (
    BRONZE_ECOMMERCE,
    BRONZE_ERP,
    BRONZE_EXTERNAL,
    BRONZE_POS,
    DELTA_PACKAGE,
    SPARK_APP_NAME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
POS_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=False),
        StructField("product_id", StringType(), nullable=False),
        StructField("store_id", StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("quantity", IntegerType(), nullable=False),
        StructField("unit_price", DoubleType(), nullable=False),
        StructField("total_amount", DoubleType(), nullable=False),
        StructField("return_flag", BooleanType(), nullable=False),
    ]
)

ECOMMERCE_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), nullable=False),
        StructField("session_id", StringType(), nullable=True),
        StructField("customer_id", StringType(), nullable=True),
        StructField("product_id", StringType(), nullable=False),
        StructField("event_type", StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("device_type", StringType(), nullable=True),
    ]
)

ERP_SCHEMA = StructType(
    [
        StructField("product_id", StringType(), nullable=False),
        StructField("store_id", StringType(), nullable=False),
        StructField("snapshot_date", DateType(), nullable=False),
        StructField("stock_quantity", IntegerType(), nullable=False),
        StructField("reorder_point", IntegerType(), nullable=True),
        StructField("lead_time_days", IntegerType(), nullable=True),
        StructField("supplier_id", StringType(), nullable=True),
    ]
)

EXTERNAL_SCHEMA = StructType(
    [
        StructField("date", DateType(), nullable=False),
        StructField("weather_temp_c", DoubleType(), nullable=True),
        StructField("weather_rain_mm", DoubleType(), nullable=True),
        StructField("cpi_index", DoubleType(), nullable=True),
        StructField("competitor_sku_id", StringType(), nullable=True),
        StructField("competitor_price", DoubleType(), nullable=True),
    ]
)


class SparkIngestor:
    """PySpark-based ingestion engine for the NeuralRetail bronze layer.

    Handles reading CSV/Parquet/JSON source files, enforcing schemas,
    enriching with metadata columns, and writing to partitioned Delta tables.

    Example:
        >>> ingestor = SparkIngestor()
        >>> spark = ingestor.get_spark_session()
        >>> ingestor.run_all({"pos": "s3://bucket/pos/", "erp": "s3://bucket/erp/"})
    """

    def get_spark_session(self) -> SparkSession:
        """Build and return a configured SparkSession with Delta Lake support.

        Returns:
            Active SparkSession with Delta Lake extensions configured.
        """
        spark = (
            SparkSession.builder.appName(SPARK_APP_NAME)
            .config("spark.jars.packages", DELTA_PACKAGE)
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark

    def _enrich_metadata(
        self, df: DataFrame, source_system: str
    ) -> DataFrame:
        """Add standard metadata columns to any ingested DataFrame.

        Args:
            df: Raw source DataFrame.
            source_system: Label for the originating system (e.g., "POS").

        Returns:
            DataFrame with ingestion_timestamp and source_system columns added.
        """
        return df.withColumn(
            "ingestion_timestamp", F.current_timestamp()
        ).withColumn("source_system", F.lit(source_system))

    def ingest_pos(self, spark: SparkSession, source_path: str) -> None:
        """Ingest POS transaction data into the bronze Delta table.

        Reads CSV/Parquet from source_path, enforces POS_SCHEMA, enriches with
        metadata, and writes partitioned Delta Lake table with mergeSchema=true.

        Args:
            spark: Active SparkSession.
            source_path: Local or cloud path to raw POS data files.
        """
        logger.info("Starting POS ingestion from: %s", source_path)

        df = (
            spark.read.format("csv")
            .option("header", "true")
            .schema(POS_SCHEMA)
            .load(source_path)
        )

        # Validate critical non-null columns
        null_count = df.filter(
            F.col("transaction_id").isNull()
            | F.col("customer_id").isNull()
            | F.col("product_id").isNull()
        ).count()

        if null_count > 0:
            raise ValueError(
                f"POS data has {null_count} rows with null key columns "
                "(transaction_id, customer_id, product_id)."
            )

        enriched = self._enrich_metadata(df, "POS")

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy(
                F.year(F.col("timestamp")).alias("year"),
                F.month(F.col("timestamp")).alias("month"),
            )
            .option("mergeSchema", "true")
            .save(BRONZE_POS)
        )

        row_count = enriched.count()
        logger.info("POS ingestion complete: %d rows written to %s", row_count, BRONZE_POS)

    def ingest_ecommerce(self, spark: SparkSession, source_path: str) -> None:
        """Ingest e-commerce clickstream events into the bronze Delta table.

        Args:
            spark: Active SparkSession.
            source_path: Local or cloud path to raw e-commerce event files.
        """
        logger.info("Starting e-commerce ingestion from: %s", source_path)

        df = (
            spark.read.format("csv")
            .option("header", "true")
            .schema(ECOMMERCE_SCHEMA)
            .load(source_path)
        )

        enriched = self._enrich_metadata(df, "ECOMMERCE")

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy(
                F.year(F.col("timestamp")).alias("year"),
                F.month(F.col("timestamp")).alias("month"),
            )
            .option("mergeSchema", "true")
            .save(BRONZE_ECOMMERCE)
        )

        row_count = enriched.count()
        logger.info(
            "E-commerce ingestion complete: %d rows written to %s",
            row_count,
            BRONZE_ECOMMERCE,
        )

    def ingest_erp(self, spark: SparkSession, source_path: str) -> None:
        """Ingest ERP inventory snapshot data into the bronze Delta table.

        Args:
            spark: Active SparkSession.
            source_path: Local or cloud path to raw ERP data files.
        """
        logger.info("Starting ERP ingestion from: %s", source_path)

        df = (
            spark.read.format("csv")
            .option("header", "true")
            .schema(ERP_SCHEMA)
            .load(source_path)
        )

        null_erp = df.filter(
            F.col("product_id").isNull()
            | F.col("store_id").isNull()
            | F.col("snapshot_date").isNull()
        ).count()

        if null_erp > 0:
            raise ValueError(
                f"ERP data has {null_erp} rows with null key columns."
            )

        enriched = self._enrich_metadata(df, "ERP")

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy("snapshot_date")
            .option("mergeSchema", "true")
            .save(BRONZE_ERP)
        )

        row_count = enriched.count()
        logger.info(
            "ERP ingestion complete: %d rows written to %s", row_count, BRONZE_ERP
        )

    def ingest_external(self, spark: SparkSession, source_path: str) -> None:
        """Ingest external signals (weather, CPI, competitor pricing) into bronze.

        Args:
            spark: Active SparkSession.
            source_path: Local or cloud path to raw external signal files.
        """
        logger.info("Starting external signals ingestion from: %s", source_path)

        df = (
            spark.read.format("csv")
            .option("header", "true")
            .schema(EXTERNAL_SCHEMA)
            .load(source_path)
        )

        enriched = self._enrich_metadata(df, "EXTERNAL")

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy("date")
            .option("mergeSchema", "true")
            .save(BRONZE_EXTERNAL)
        )

        row_count = enriched.count()
        logger.info(
            "External ingestion complete: %d rows written to %s",
            row_count,
            BRONZE_EXTERNAL,
        )

    def run_all(self, source_paths: dict[str, str]) -> dict[str, Any]:
        """Orchestrate ingestion across all four source systems.

        Args:
            source_paths: Mapping of system name to source path.
                Expected keys: "pos", "ecommerce", "erp", "external".

        Returns:
            Dict mapping system name to row count ingested.

        Raises:
            KeyError: If a required source path key is missing.
        """
        spark = self.get_spark_session()
        results: dict[str, Any] = {}

        ingestors = {
            "pos": self.ingest_pos,
            "ecommerce": self.ingest_ecommerce,
            "erp": self.ingest_erp,
            "external": self.ingest_external,
        }

        for system, ingest_fn in ingestors.items():
            if system not in source_paths:
                logger.warning("No source path provided for system: %s — skipping", system)
                continue
            try:
                ingest_fn(spark, source_paths[system])
                results[system] = "SUCCESS"
                logger.info("[%s] ingestion succeeded", system.upper())
            except Exception as exc:
                logger.error("[%s] ingestion FAILED: %s", system.upper(), exc)
                results[system] = f"FAILED: {exc}"

        logger.info("run_all summary: %s", results)
        return results
