"""NeuralRetail Dashboard — Chart Builder Utilities.

Day 16 — NeuralRetail AMX-DS-2026-04
Centralised Plotly chart constructors shared across all dashboard pages.
Applies NeuralRetail brand template (E84E1B / F7941D / FBBA13) consistently.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------
PRIMARY = "#E84E1B"
SECONDARY = "#F7941D"
ACCENT = "#FBBA13"
CI_FILL = "rgba(251,186,19,0.20)"
GRID_COLOR = "#f5f5f5"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"


class ChartBuilder:
    """Centralised Plotly chart factory with NeuralRetail brand template.

    All methods return :class:`plotly.graph_objects.Figure` instances with
    consistent styling applied via :meth:`apply_brand_template`.

    Usage::

        cb = ChartBuilder()
        fig = cb.build_forecast_chart(demand_df, sku_id="SKU-1001")
        st.plotly_chart(fig, use_container_width=True)
    """

    # ------------------------------------------------------------------
    # Forecast chart
    # ------------------------------------------------------------------

    def build_forecast_chart(self, df: pd.DataFrame, sku_id: str) -> go.Figure:
        """Build actual + P50 forecast with P10/P90 confidence band.

        Args:
            df: DataFrame with columns [date, actual, p10, p50, p90, is_forecast].
                ``actual`` is NaN for forecast rows; ``p10/p50/p90`` are NaN
                for historical rows.
            sku_id: SKU identifier used in chart title.

        Returns:
            Plotly Figure with rangeslider, brand colours, and hover unification.
        """
        fig = go.Figure()

        hist_df = df[~df.get("is_forecast", pd.Series(False, index=df.index))]
        fore_df = df[df.get("is_forecast", pd.Series(False, index=df.index))]

        # CI band
        if not fore_df.empty and "p10" in df.columns and "p90" in df.columns:
            dates_fwd = list(fore_df["date"]) + list(fore_df["date"])[::-1]
            y_band = list(fore_df["p90"]) + list(fore_df["p10"])[::-1]
            fig.add_trace(
                go.Scatter(
                    x=dates_fwd, y=y_band,
                    fill="toself", fillcolor=CI_FILL,
                    line=dict(color="rgba(0,0,0,0)"),
                    name="P10–P90 CI", hoverinfo="skip",
                )
            )

        # Actual
        if not hist_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_df["date"], y=hist_df["actual"],
                    name="Actual", line=dict(color=PRIMARY, width=2.5),
                    mode="lines",
                )
            )

        # P50 forecast
        if not fore_df.empty and "p50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=fore_df["date"], y=fore_df["p50"],
                    name="Forecast P50",
                    line=dict(color=SECONDARY, width=2.5, dash="dash"),
                )
            )

        # Forecast start marker
        if not fore_df.empty:
            fig.add_vline(
                x=fore_df["date"].min(), line_dash="dot", line_color="#ccc",
                annotation_text="Forecast →", annotation_position="top right",
            )

        fig.update_layout(
            title=dict(text=f"Demand Forecast — {sku_id}", font=dict(size=15)),
            xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
            hovermode="x unified",
        )
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # MAPE leaderboard
    # ------------------------------------------------------------------

    def build_mape_leaderboard(self, df: pd.DataFrame) -> go.Figure:
        """Build horizontal MAPE leaderboard bar chart with conditional coloring.

        Green ≤ 10% | Amber 10-15% | Red > 15%.

        Args:
            df: DataFrame with columns [sku_id, mape] at minimum.

        Returns:
            Plotly Figure.
        """
        df_sorted = df.sort_values("mape") if "mape" in df.columns else df
        mape_col = df_sorted.get("mape", pd.Series([0.0] * len(df_sorted)))
        colors = [
            "#16a34a" if m <= 10 else SECONDARY if m <= 15 else PRIMARY
            for m in mape_col
        ]
        sku_col = df_sorted.get("sku_id", pd.Series([f"SKU-{i}" for i in range(len(df_sorted))]))

        fig = go.Figure(
            go.Bar(
                x=mape_col, y=sku_col, orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.04)", width=1)),
                text=[f"{m:.1f}%" for m in mape_col], textposition="outside",
                hovertemplate="%{y} — MAPE: %{x:.1f}%<extra></extra>",
            )
        )
        fig.add_vline(x=10.0, line_dash="dash", line_color=PRIMARY, annotation_text="Target 10%")
        fig.add_vline(x=15.0, line_dash="dot", line_color="#dc2626", annotation_text="Danger 15%")
        fig.update_layout(
            title="SKU MAPE Leaderboard",
            xaxis=dict(title="MAPE (%)", range=[0, mape_col.max() * 1.2 + 2]),
        )
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # Segment radar
    # ------------------------------------------------------------------

    def build_segment_radar(
        self,
        profiles_df: pd.DataFrame,
        selected_segments: list[str],
        axes: list[str] | None = None,
    ) -> go.Figure:
        """Build RFM segment polar radar chart.

        Args:
            profiles_df: DataFrame indexed by ``persona`` with numeric RFM columns.
            selected_segments: Persona names to overlay.
            axes: Column names to use as radar axes. Defaults to all numeric columns.

        Returns:
            Plotly Figure.
        """
        if axes is None:
            axes = [c for c in profiles_df.columns if c != "persona"]

        color_cycle = [PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7", "#94A3B8"]
        fig = go.Figure()

        for i, persona in enumerate(selected_segments):
            row = profiles_df[profiles_df.get("persona", profiles_df.index) == persona]
            if row.empty:
                continue
            values = [float(row[ax].iloc[0]) for ax in axes if ax in row.columns]
            values.append(values[0])
            ax_labels = [ax for ax in axes if ax in row.columns] + [axes[0] if axes else ""]
            color = color_cycle[i % len(color_cycle)]
            fig.add_trace(
                go.Scatterpolar(
                    r=values, theta=ax_labels, fill="toself",
                    name=persona,
                    line=dict(color=color, width=2.5),
                    fillcolor=f"{color}22",
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#eee"),
                angularaxis=dict(gridcolor="#eee"),
            ),
            title="Segment RFM Radar",
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
        )
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # Churn heatmap
    # ------------------------------------------------------------------

    def build_churn_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Build churn risk density heatmap (segment × risk decile).

        Args:
            df: DataFrame with columns [persona, risk_decile, customer_count].

        Returns:
            Plotly Figure.
        """
        import plotly.express as px

        fig = px.density_heatmap(
            df,
            x="persona" if "persona" in df.columns else df.columns[0],
            y="risk_decile" if "risk_decile" in df.columns else df.columns[1],
            z="customer_count" if "customer_count" in df.columns else df.columns[2],
            color_continuous_scale=[[0, "#DCFCE7"], [0.5, ACCENT], [1.0, PRIMARY]],
            title="Churn Risk Heatmap",
            text_auto=True,
        )
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # Inventory scatter
    # ------------------------------------------------------------------

    def build_inventory_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Build overstock risk quadrant scatter with quadrant lines.

        Args:
            df: DataFrame with columns [sku_id, days_of_supply, holding_cost,
                overstock_units, dead_stock_risk, category].

        Returns:
            Plotly Figure.
        """
        import plotly.express as px

        fig = px.scatter(
            df,
            x="days_of_supply",
            y="holding_cost",
            size="overstock_units" if "overstock_units" in df.columns else None,
            color="dead_stock_risk" if "dead_stock_risk" in df.columns else None,
            color_continuous_scale=[[0, "#DCFCE7"], [0.5, ACCENT], [1.0, PRIMARY]],
            hover_data=["sku_id"] if "sku_id" in df.columns else None,
            title="Overstock Risk Analysis",
        )
        fig.add_vline(x=90, line_dash="dash", line_color="#aaa", annotation_text="90 days")
        fig.add_hline(y=10_000, line_dash="dash", line_color="#aaa", annotation_text="£10k")
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # Drift PSI bars
    # ------------------------------------------------------------------

    def build_drift_bars(self, psi_df: pd.DataFrame) -> go.Figure:
        """Build PSI drift indicator horizontal bar chart.

        Args:
            psi_df: DataFrame with columns [feature, psi] at minimum.

        Returns:
            Plotly Figure with 0.1 and 0.2 threshold lines.
        """
        psi_col = psi_df["psi"] if "psi" in psi_df.columns else psi_df.iloc[:, 1]
        feat_col = psi_df["feature"] if "feature" in psi_df.columns else psi_df.iloc[:, 0]
        colors = [
            PRIMARY if p > 0.2 else SECONDARY if p > 0.1 else "#16a34a"
            for p in psi_col
        ]
        fig = go.Figure(
            go.Bar(
                x=psi_col, y=feat_col, orientation="h",
                marker=dict(color=colors),
                text=[f"{p:.3f}" for p in psi_col], textposition="outside",
                hovertemplate="%{y}: PSI=%{x:.4f}<extra></extra>",
            )
        )
        fig.add_vline(x=0.1, line_dash="dash", line_color=SECONDARY, annotation_text="Moderate")
        fig.add_vline(x=0.2, line_dash="dash", line_color=PRIMARY, annotation_text="Severe")
        fig.update_layout(title="Feature Drift PSI Dashboard")
        return self.apply_brand_template(fig)

    # ------------------------------------------------------------------
    # Brand template
    # ------------------------------------------------------------------

    def apply_brand_template(self, fig: go.Figure) -> go.Figure:
        """Apply NeuralRetail brand layout to any Plotly Figure.

        Standardises font, background, grid colour, and margins across all
        charts to ensure visual consistency.

        Args:
            fig: Any Plotly Figure object.

        Returns:
            The same Figure with updated layout properties.
        """
        fig.update_layout(
            font=dict(family=FONT_FAMILY, size=12, color="#1a1a1a"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(
                gridcolor=GRID_COLOR,
                zeroline=False,
                linecolor="#e5e5e5",
            ),
            yaxis=dict(
                gridcolor=GRID_COLOR,
                zeroline=False,
                linecolor="#e5e5e5",
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#f0f0f0",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#ddd",
                font=dict(family=FONT_FAMILY, size=12),
            ),
            colorway=[PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7", "#94A3B8"],
        )
        return fig
