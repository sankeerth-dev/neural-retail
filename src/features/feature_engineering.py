"""Polars-based feature engineering for the NeuralRetail silver layer.

Computes RFM customer features, SKU demand time-series features, calendar/date
features, and joins external signals (weather, CPI) for model training.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

HOLIDAY_CALENDAR_PATH = Path("configs/holiday_calendar.json")


def _load_holidays() -> list[date]:
    """Load holiday dates from the NeuralRetail holiday calendar JSON.

    Returns:
        List of date objects representing retail holiday dates.
    """
    if not HOLIDAY_CALENDAR_PATH.exists():
        logger.warning("Holiday calendar not found at %s", HOLIDAY_CALENDAR_PATH)
        return []
    with open(HOLIDAY_CALENDAR_PATH) as f:
        entries = json.load(f)
    return [date.fromisoformat(e["date"]) for e in entries]


class FeatureEngineer:
    """Polars-based feature engineering engine for NeuralRetail silver layer.

    Computes RFM, demand, date, and external feature sets from bronze-layer
    transaction data. All methods return new Polars DataFrames without
    mutating the input.

    Example:
        >>> fe = FeatureEngineer()
        >>> rfm_df = fe.compute_rfm(txn_df, snapshot_date=date(2026, 5, 1))
        >>> demand_df = fe.compute_demand_features(daily_sales_df)
    """

    def __init__(self) -> None:
        """Initialise the FeatureEngineer and pre-load holiday calendar."""
        self._holidays: list[date] = _load_holidays()

    def compute_rfm(
        self,
        df: pl.DataFrame,
        snapshot_date: date,
    ) -> pl.DataFrame:
        """Compute RFM (Recency, Frequency, Monetary) features per customer.

        Uses a 90-day lookback window from snapshot_date. RFM scores are
        min-max normalised and combined: rfm_score = recency*0.30 + freq*0.35 + monetary*0.35.

        Args:
            df: Polars DataFrame with columns:
                customer_id (str), timestamp (datetime), total_amount (float).
            snapshot_date: Reference date for recency calculation.

        Returns:
            Polars DataFrame with one row per customer and columns:
            customer_id, recency_days, frequency, monetary,
            avg_basket_size, rfm_score.
        """
        snap = pl.lit(snapshot_date).cast(pl.Date)
        window_start = snapshot_date - timedelta(days=90)

        windowed = df.filter(
            pl.col("timestamp").cast(pl.Date) >= window_start
        )

        rfm = (
            windowed.group_by("customer_id")
            .agg(
                [
                    (
                        snap
                        - pl.col("timestamp").cast(pl.Date).max()
                    )
                    .dt.total_days()
                    .cast(pl.Int64)
                    .alias("recency_days"),
                    pl.col("timestamp").count().alias("frequency"),
                    pl.col("total_amount").sum().alias("monetary"),
                ]
            )
            .with_columns(
                (pl.col("monetary") / pl.col("frequency")).alias("avg_basket_size")
            )
        )

        # Min-max normalise 1–5
        def _norm(col_name: str, invert: bool = False) -> pl.Expr:
            col = pl.col(col_name)
            min_val = col.min()
            max_val = col.max()
            normed = (col - min_val) / (max_val - min_val + 1e-9)
            normed = normed * 4 + 1  # Scale to 1-5
            if invert:
                normed = 6 - normed  # Higher recency_days → lower score
            return normed

        rfm = rfm.with_columns(
            [
                _norm("recency_days", invert=True).alias("recency_norm"),
                _norm("frequency").alias("frequency_norm"),
                _norm("monetary").alias("monetary_norm"),
            ]
        ).with_columns(
            (
                pl.col("recency_norm") * 0.30
                + pl.col("frequency_norm") * 0.35
                + pl.col("monetary_norm") * 0.35
            ).alias("rfm_score")
        )

        return rfm.select(
            [
                "customer_id",
                "recency_days",
                "frequency",
                "monetary",
                "avg_basket_size",
                "rfm_score",
            ]
        )

    def compute_demand_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute SKU-level time-series demand features.

        Expects daily aggregated sales with columns product_id and date and quantity.
        Returns rolling statistics, lag features, and momentum.

        Args:
            df: Polars DataFrame with columns:
                product_id (str), date (date), quantity (int/float).

        Returns:
            DataFrame with all original columns plus: rolling_mean_7d,
            rolling_mean_14d, rolling_mean_30d, rolling_std_7d, rolling_std_14d,
            lag_1d, lag_7d, lag_14d, momentum_7d.
        """
        df = df.sort(["product_id", "date"])

        result = df.with_columns(
            [
                pl.col("quantity")
                .rolling_mean(window_size=7)
                .over("product_id")
                .alias("rolling_mean_7d"),
                pl.col("quantity")
                .rolling_mean(window_size=14)
                .over("product_id")
                .alias("rolling_mean_14d"),
                pl.col("quantity")
                .rolling_mean(window_size=30)
                .over("product_id")
                .alias("rolling_mean_30d"),
                pl.col("quantity")
                .rolling_std(window_size=7)
                .over("product_id")
                .alias("rolling_std_7d"),
                pl.col("quantity")
                .rolling_std(window_size=14)
                .over("product_id")
                .alias("rolling_std_14d"),
                pl.col("quantity")
                .shift(1)
                .over("product_id")
                .alias("lag_1d"),
                pl.col("quantity")
                .shift(7)
                .over("product_id")
                .alias("lag_7d"),
                pl.col("quantity")
                .shift(14)
                .over("product_id")
                .alias("lag_14d"),
            ]
        ).with_columns(
            (
                (pl.col("quantity") - pl.col("rolling_mean_7d"))
                / (pl.col("rolling_mean_7d") + 1e-9)
            ).alias("momentum_7d")
        )

        return result

    def compute_date_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute calendar and promotional date features.

        Args:
            df: Polars DataFrame with a ``date`` column (pl.Date).

        Returns:
            DataFrame with calendar feature columns appended:
            day_of_week, week_of_year, month, quarter, is_weekend,
            is_month_end, is_quarter_end, days_to_next_holiday,
            days_since_last_holiday, is_promotional_period.
        """
        holidays = self._holidays

        def _days_to_next_holiday(d: date) -> int:
            future = [h for h in holidays if h >= d]
            return (min(future) - d).days if future else 999

        def _days_since_last_holiday(d: date) -> int:
            past = [h for h in holidays if h <= d]
            return (d - max(past)).days if past else 999

        result = df.with_columns(
            [
                pl.col("date").dt.weekday().alias("day_of_week"),
                pl.col("date").dt.week().alias("week_of_year"),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.quarter().alias("quarter"),
                (pl.col("date").dt.weekday() >= 5).alias("is_weekend"),
                (pl.col("date").dt.month_end() == pl.col("date")).alias("is_month_end"),
                (
                    (pl.col("date").dt.month().is_in([3, 6, 9, 12]))
                    & (pl.col("date").dt.month_end() == pl.col("date"))
                ).alias("is_quarter_end"),
            ]
        )

        # Apply Python UDFs for holiday proximity features
        if holidays:
            dates_list = result["date"].to_list()
            days_to_next = [_days_to_next_holiday(d) for d in dates_list]
            days_since_last = [_days_since_last_holiday(d) for d in dates_list]
            is_promo = [d <= 7 or n <= 7 for d, n in zip(days_since_last, days_to_next)]

            result = result.with_columns(
                [
                    pl.Series("days_to_next_holiday", days_to_next, dtype=pl.Int32),
                    pl.Series("days_since_last_holiday", days_since_last, dtype=pl.Int32),
                    pl.Series("is_promotional_period", is_promo, dtype=pl.Boolean),
                ]
            )
        else:
            result = result.with_columns(
                [
                    pl.lit(999).cast(pl.Int32).alias("days_to_next_holiday"),
                    pl.lit(999).cast(pl.Int32).alias("days_since_last_holiday"),
                    pl.lit(False).alias("is_promotional_period"),
                ]
            )

        return result

    def join_external(
        self,
        demand_df: pl.DataFrame,
        weather_df: pl.DataFrame,
        cpi_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join external weather and CPI signals onto demand features.

        Weather join: demand LEFT JOIN weather ON (store_id, date).
        CPI join: result LEFT JOIN cpi ON (category, month).

        Args:
            demand_df: SKU demand DataFrame with store_id and date.
            weather_df: Weather DataFrame with store_id, date, temp_c, rain_mm.
            cpi_df: CPI DataFrame with category, month (int), cpi_index, cpi_mom_change.

        Returns:
            Joined Polars DataFrame with all demand features plus weather
            and CPI columns.
        """
        result = demand_df.join(
            weather_df.select(["store_id", "date", "temp_c", "rain_mm", "is_extreme_weather"]),
            on=["store_id", "date"],
            how="left",
        )

        if "month" not in result.columns:
            result = result.with_columns(
                pl.col("date").dt.month().cast(pl.Int32).alias("month")
            )

        if "category" in result.columns and "category" in cpi_df.columns:
            result = result.join(
                cpi_df.select(["category", "month", "cpi_index", "cpi_mom_change"]),
                on=["category", "month"],
                how="left",
            )

        return result
