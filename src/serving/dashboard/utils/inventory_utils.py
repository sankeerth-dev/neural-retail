"""NeuralRetail Dashboard — Inventory Utilities.

Day 18 — NeuralRetail AMX-DS-2026-04
ABC/XYZ classification, EOQ computation, dead-stock scoring,
and reorder alert generation for the Inventory Health dashboard.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InventoryUtils:
    """Inventory intelligence utility class.

    Provides ABC-XYZ classification, Economic Order Quantity calculations,
    dead-stock scoring, and dynamic reorder alert generation.

    All methods operate on pandas DataFrames and return pandas DataFrames,
    making them composable with Streamlit's data display components.
    """

    # ------------------------------------------------------------------
    # ABC classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_abc(
        df: pd.DataFrame,
        revenue_col: str = "total_revenue",
        sku_col: str = "sku_id",
        a_threshold: float = 0.70,
        b_threshold: float = 0.90,
    ) -> pd.DataFrame:
        """Classify SKUs into ABC tiers by cumulative revenue contribution.

        - A: top ``a_threshold`` (default 70%) of cumulative revenue.
        - B: ``a_threshold`` to ``b_threshold`` (70-90%).
        - C: remaining (90-100%).

        Args:
            df: DataFrame with at least ``sku_col`` and ``revenue_col``.
            revenue_col: Column name containing revenue values (must be numeric).
            sku_col: Column name for SKU identifiers.
            a_threshold: Cumulative revenue fraction for class A boundary.
            b_threshold: Cumulative revenue fraction for class B boundary.

        Returns:
            Input DataFrame with an additional ``abc_class`` column (str: A/B/C)
            and ``revenue_pct`` (float) column added. Sorted by revenue descending.

        Raises:
            KeyError: If ``sku_col`` or ``revenue_col`` are not in ``df``.
            ValueError: If thresholds are not in (0, 1) with a_threshold < b_threshold.
        """
        if revenue_col not in df.columns:
            raise KeyError(f"Revenue column '{revenue_col}' not found in DataFrame.")
        if sku_col not in df.columns:
            raise KeyError(f"SKU column '{sku_col}' not found in DataFrame.")
        if not (0 < a_threshold < b_threshold < 1):
            raise ValueError("Thresholds must satisfy 0 < a_threshold < b_threshold < 1.")

        df_sorted = df.sort_values(revenue_col, ascending=False).copy()
        total_revenue = df_sorted[revenue_col].sum()
        if total_revenue == 0:
            df_sorted["abc_class"] = "C"
            df_sorted["revenue_pct"] = 0.0
            return df_sorted

        df_sorted["revenue_pct"] = df_sorted[revenue_col] / total_revenue
        df_sorted["cumulative_pct"] = df_sorted["revenue_pct"].cumsum()

        def _assign_abc(cum_pct: float) -> str:
            if cum_pct <= a_threshold:
                return "A"
            elif cum_pct <= b_threshold:
                return "B"
            return "C"

        df_sorted["abc_class"] = df_sorted["cumulative_pct"].apply(_assign_abc)
        df_sorted = df_sorted.drop(columns=["cumulative_pct"])
        logger.info(
            "ABC classification: A=%d  B=%d  C=%d",
            (df_sorted["abc_class"] == "A").sum(),
            (df_sorted["abc_class"] == "B").sum(),
            (df_sorted["abc_class"] == "C").sum(),
        )
        return df_sorted

    # ------------------------------------------------------------------
    # XYZ classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_xyz(
        df: pd.DataFrame,
        demand_col: str = "demand",
        sku_col: str = "sku_id",
        date_col: str = "date",
        x_cv_threshold: float = 0.50,
        y_cv_threshold: float = 1.00,
    ) -> pd.DataFrame:
        """Classify SKUs by demand variability using Coefficient of Variation.

        - X: CV < ``x_cv_threshold`` (0.5) — highly predictable.
        - Y: ``x_cv_threshold`` ≤ CV < ``y_cv_threshold`` (0.5-1.0) — moderate.
        - Z: CV ≥ ``y_cv_threshold`` (≥1.0) — highly variable/unpredictable.

        Args:
            df: DataFrame in long format with columns [sku_col, date_col, demand_col].
            demand_col: Column name for demand values.
            sku_col: Column name for SKU identifiers.
            date_col: Column name for date dimension (used for grouping only).
            x_cv_threshold: CV boundary between X and Y.
            y_cv_threshold: CV boundary between Y and Z.

        Returns:
            DataFrame with one row per SKU, columns: [sku_id, mean_demand,
            std_demand, cv, xyz_class].
        """
        grouped = df.groupby(sku_col)[demand_col].agg(
            mean_demand="mean",
            std_demand="std",
        ).reset_index()
        grouped["std_demand"] = grouped["std_demand"].fillna(0.0)
        grouped["cv"] = grouped.apply(
            lambda r: r["std_demand"] / r["mean_demand"] if r["mean_demand"] > 0 else 0.0,
            axis=1,
        )

        def _assign_xyz(cv: float) -> str:
            if cv < x_cv_threshold:
                return "X"
            elif cv < y_cv_threshold:
                return "Y"
            return "Z"

        grouped["xyz_class"] = grouped["cv"].apply(_assign_xyz)
        logger.info(
            "XYZ classification: X=%d  Y=%d  Z=%d",
            (grouped["xyz_class"] == "X").sum(),
            (grouped["xyz_class"] == "Y").sum(),
            (grouped["xyz_class"] == "Z").sum(),
        )
        return grouped

    # ------------------------------------------------------------------
    # EOQ computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_eoq(
        annual_demand: float,
        order_cost: float,
        holding_cost_pct: float,
        unit_cost: float,
        lead_time_days: int = 14,
        safety_factor: float = 1.65,
        demand_std_daily: float | None = None,
    ) -> dict[str, float]:
        """Compute Economic Order Quantity and related inventory parameters.

        Uses the classic Wilson EOQ formula:
        ``EOQ = sqrt(2 * D * S / H)``
        where D = annual demand, S = order cost, H = annual holding cost per unit.

        Safety stock: ``SS = Z * σ_d * sqrt(LT)``
        Reorder point: ``ROP = d_avg * LT + SS``

        Args:
            annual_demand: Annual demand in units (D).
            order_cost: Fixed cost per purchase order in £ (S).
            holding_cost_pct: Annual holding cost as a percentage of unit cost (h).
                Converted to H = h/100 * unit_cost.
            unit_cost: Per-unit purchase cost in £.
            lead_time_days: Supplier lead time in days.
            safety_factor: Z-score for desired service level (1.28=90%, 1.65=95%,
                2.05=98%, 2.33=99%).
            demand_std_daily: Daily demand standard deviation. If None, estimated
                as ``sqrt(annual_demand / 365)``.

        Returns:
            Dict with keys:
            - ``eoq``: Optimal order quantity (units).
            - ``safety_stock``: Safety stock (units).
            - ``reorder_point``: Reorder trigger point (units).
            - ``total_annual_cost``: Purchase + ordering + holding cost (£).
            - ``orders_per_year``: Number of purchase orders per year.
            - ``avg_cycle_stock``: Average cycle stock = EOQ / 2 (units).
        """
        H = (holding_cost_pct / 100.0) * unit_cost
        if H <= 0 or annual_demand <= 0 or order_cost <= 0:
            logger.warning("Invalid EOQ inputs: D=%s S=%s H=%s", annual_demand, order_cost, H)
            zero_result: dict[str, float] = {
                "eoq": 0.0, "safety_stock": 0.0, "reorder_point": 0.0,
                "total_annual_cost": 0.0, "orders_per_year": 0.0, "avg_cycle_stock": 0.0,
            }
            return zero_result

        eoq = math.sqrt((2.0 * annual_demand * order_cost) / H)
        daily_demand = annual_demand / 365.0
        if demand_std_daily is None:
            demand_std_daily = math.sqrt(max(daily_demand, 1.0))

        safety_stock = safety_factor * demand_std_daily * math.sqrt(lead_time_days)
        reorder_point = daily_demand * lead_time_days + safety_stock
        orders_per_year = annual_demand / eoq
        avg_cycle_stock = eoq / 2.0
        ordering_cost_annual = orders_per_year * order_cost
        holding_cost_annual = (avg_cycle_stock + safety_stock) * H
        purchase_cost_annual = annual_demand * unit_cost
        total_annual_cost = purchase_cost_annual + ordering_cost_annual + holding_cost_annual

        return {
            "eoq": round(eoq, 0),
            "safety_stock": round(safety_stock, 0),
            "reorder_point": round(reorder_point, 0),
            "total_annual_cost": round(total_annual_cost, 2),
            "orders_per_year": round(orders_per_year, 1),
            "avg_cycle_stock": round(avg_cycle_stock, 0),
        }

    # ------------------------------------------------------------------
    # Dead stock scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score_dead_stock(
        df: pd.DataFrame,
        sku_col: str = "sku_id",
        days_since_sale_col: str = "days_since_last_sale",
        lead_time_col: str = "avg_lead_time",
        threshold_days: int = 180,
    ) -> pd.DataFrame:
        """Flag and score dead-stock SKUs by days since last sale.

        Dead stock is defined as items where ``days_since_last_sale > threshold_days``.
        The dead-stock score is: ``days_since_last_sale / avg_lead_time``, capped at 50.
        Higher scores indicate stock that is increasingly unlikely to sell before
        the next replenishment cycle.

        Args:
            df: DataFrame containing at least ``sku_col``, ``days_since_sale_col``,
                and ``lead_time_col``.
            sku_col: Column name for SKU identifiers.
            days_since_sale_col: Column name for days since last sale.
            lead_time_col: Column name for average supplier lead time in days.
            threshold_days: Minimum days-since-sale to flag as dead stock.

        Returns:
            Input DataFrame with two additional columns:
            - ``dead_stock_flag`` (bool): True if days_since_sale > threshold.
            - ``dead_stock_score`` (float): Score = days_since_sale / avg_lead_time,
              capped at 50. Higher is worse.
        """
        df = df.copy()
        if days_since_sale_col not in df.columns:
            raise KeyError(f"Column '{days_since_sale_col}' not found.")
        if lead_time_col not in df.columns:
            logger.warning("Lead time column '%s' not found; defaulting to 14 days.", lead_time_col)
            df[lead_time_col] = 14.0

        df["dead_stock_flag"] = df[days_since_sale_col] > threshold_days
        df["dead_stock_score"] = (
            df[days_since_sale_col] / df[lead_time_col].clip(lower=1.0)
        ).clip(upper=50.0).round(2)

        n_dead = df["dead_stock_flag"].sum()
        logger.info(
            "Dead stock: %d / %d SKUs flagged (threshold=%dd)",
            n_dead, len(df), threshold_days,
        )
        return df

    # ------------------------------------------------------------------
    # Reorder alert generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_reorder_alerts(
        inventory_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        sku_col: str = "sku_id",
        stock_col: str = "current_stock",
        reorder_point_col: str = "reorder_point",
        avg_demand_col: str = "avg_daily_demand",
        critical_days: int = 7,
        high_days: int = 14,
        medium_days: int = 30,
    ) -> pd.DataFrame:
        """Generate urgency-classified reorder alerts for all SKUs.

        Computes ``days_until_stockout`` from current stock and average daily
        demand, then classifies urgency:
        - Critical: ≤ ``critical_days`` (default 7).
        - High: ≤ ``high_days`` (default 14).
        - Medium: ≤ ``medium_days`` (default 30).
        - OK: > ``medium_days``.

        Args:
            inventory_df: DataFrame with columns [sku_id, current_stock,
                reorder_point]. Extra columns are preserved in output.
            forecast_df: Demand forecast DataFrame with [sku_id, p50] or
                [sku_id, avg_daily_demand]. Used to extract average daily
                demand for each SKU.
            sku_col: Column name for SKU identifiers.
            stock_col: Column name for current inventory level.
            reorder_point_col: Column name for the reorder trigger level.
            avg_demand_col: Column name in ``forecast_df`` for average daily demand.
            critical_days: Days-until-stockout threshold for Critical urgency.
            high_days: Threshold for High urgency.
            medium_days: Threshold for Medium urgency.

        Returns:
            DataFrame with all original columns plus:
            - ``avg_daily_demand`` (float): from forecast_df or estimated.
            - ``days_until_stockout`` (float): current_stock / avg_daily_demand.
            - ``urgency`` (str): Critical / High / Medium / OK.
            - ``below_reorder_point`` (bool): True if current_stock < reorder_point.
            Sorted by urgency (Critical first) then days_until_stockout ascending.
        """
        df = inventory_df.copy()

        # Merge average daily demand from forecast
        if avg_demand_col in forecast_df.columns and sku_col in forecast_df.columns:
            demand_lookup = forecast_df[[sku_col, avg_demand_col]].drop_duplicates(sku_col)
            df = df.merge(demand_lookup, on=sku_col, how="left")
        elif "p50" in forecast_df.columns and sku_col in forecast_df.columns:
            # Compute avg daily demand from P50 forecast (assumes horizon rows)
            demand_agg = forecast_df.groupby(sku_col)["p50"].mean().reset_index()
            demand_agg.columns = [sku_col, avg_demand_col]
            df = df.merge(demand_agg, on=sku_col, how="left")
        else:
            logger.warning("Cannot extract demand from forecast_df; defaulting avg_daily_demand=10.")
            df[avg_demand_col] = 10.0

        df[avg_demand_col] = df[avg_demand_col].fillna(10.0).clip(lower=0.1)
        df["days_until_stockout"] = (
            df[stock_col] / df[avg_demand_col]
        ).round(1)

        def _urgency(days: float) -> str:
            if days <= critical_days:
                return "Critical"
            elif days <= high_days:
                return "High"
            elif days <= medium_days:
                return "Medium"
            return "OK"

        df["urgency"] = df["days_until_stockout"].apply(_urgency)

        if reorder_point_col in df.columns:
            df["below_reorder_point"] = df[stock_col] < df[reorder_point_col]
        else:
            df["below_reorder_point"] = False

        urgency_order = {"Critical": 0, "High": 1, "Medium": 2, "OK": 3}
        df["_urgency_rank"] = df["urgency"].map(urgency_order)
        df = df.sort_values(["_urgency_rank", "days_until_stockout"]).drop(columns=["_urgency_rank"])

        n_critical = (df["urgency"] == "Critical").sum()
        n_high = (df["urgency"] == "High").sum()
        logger.info("Reorder alerts: Critical=%d  High=%d", n_critical, n_high)
        return df.reset_index(drop=True)
