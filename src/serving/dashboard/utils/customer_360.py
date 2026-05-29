"""NeuralRetail Dashboard — Customer 360 Builder.

Day 17 — NeuralRetail AMX-DS-2026-04
Constructs Customer 360 views: purchase timeline chart, SHAP waterfall
image bytes, retention action formatting, and segment badge HTML.
"""

from __future__ import annotations

import io
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
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# ---------------------------------------------------------------------------
# Persona → CSS badge class mapping
# ---------------------------------------------------------------------------
_PERSONA_BADGE_MAP: dict[str, tuple[str, str]] = {
    "Champion": ("#FEF3C7", "#92400E"),
    "Loyal": ("#DCFCE7", "#166534"),
    "Potential Loyalist": ("#DBEAFE", "#1E40AF"),
    "At-Risk VIP": ("#FEE2E2", "#991B1B"),
    "New": ("#E0F2FE", "#075985"),
    "Lost": ("#F3F4F6", "#374151"),
}


class Customer360Builder:
    """Builds all visualisations and formatted outputs for the Customer 360 view.

    Each method is stateless and accepts only the data it needs.
    Intended for use within the Streamlit Customer Hub page.
    """

    # ------------------------------------------------------------------
    # Purchase timeline
    # ------------------------------------------------------------------

    def build_purchase_timeline(
        self,
        customer_id: str,
        txn_df: pd.DataFrame,
        category_col: str = "product_category",
        amount_col: str = "total_amount",
        timestamp_col: str = "timestamp",
        basket_col: str | None = "basket_size",
    ) -> go.Figure:
        """Build an interactive purchase timeline scatter chart.

        Each point represents a transaction. Size encodes basket size (if
        available); colour encodes product category.

        Args:
            customer_id: Customer identifier for chart title.
            txn_df: Transaction DataFrame. Must include ``timestamp_col``
                and ``amount_col``. Other columns are used for hover data.
            category_col: Column name for product category colour encoding.
            amount_col: Column name for Y-axis (transaction value).
            timestamp_col: Column name for X-axis (transaction date/time).
            basket_col: Column name for marker size encoding. Pass ``None``
                to use a fixed marker size.

        Returns:
            Plotly Figure with one trace per product category.
        """
        category_colors = {
            "Electronics": PRIMARY,
            "Apparel": SECONDARY,
            "Food": "#22C55E",
            "Health": "#885CF7",
            "Home": "#0EA5E9",
            "Sports": ACCENT,
        }

        fig = go.Figure()

        if txn_df.empty:
            fig.add_annotation(text="No transaction history available.", showarrow=False, font=dict(size=14))
        else:
            groups = txn_df.groupby(category_col) if category_col in txn_df.columns else [("All", txn_df)]
            for cat, cat_df in groups:
                color = category_colors.get(str(cat), "#888")
                sizes = (cat_df[basket_col] * 3 + 6).tolist() if basket_col and basket_col in cat_df.columns else 10
                hover_extras = ""
                if basket_col and basket_col in cat_df.columns:
                    custom_data = cat_df[basket_col]
                    hover_extras = "<br>Basket: %{customdata} items"
                else:
                    custom_data = None

                trace_kwargs: dict[str, Any] = dict(
                    x=cat_df[timestamp_col] if timestamp_col in cat_df.columns else cat_df.index,
                    y=cat_df[amount_col] if amount_col in cat_df.columns else cat_df.iloc[:, 0],
                    mode="markers",
                    name=str(cat),
                    marker=dict(
                        color=color,
                        size=sizes,
                        line=dict(color="white", width=1.5),
                        opacity=0.85,
                    ),
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Amount: £%{y:.2f}" + hover_extras + "<extra></extra>"
                    ),
                )
                if custom_data is not None:
                    trace_kwargs["customdata"] = custom_data

                fig.add_trace(go.Scatter(**trace_kwargs))

        fig.update_layout(
            title=dict(text=f"Purchase Timeline — {customer_id}", font=dict(size=14)),
            xaxis=dict(title="Purchase Date", gridcolor="#f5f5f5", zeroline=False),
            yaxis=dict(title="Transaction Amount (£)", gridcolor="#f5f5f5", zeroline=False),
            font=dict(family=FONT_FAMILY, size=11),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=50, r=20, t=50, b=70),
            hovermode="closest",
        )
        return fig

    # ------------------------------------------------------------------
    # SHAP waterfall image
    # ------------------------------------------------------------------

    def build_shap_waterfall_image(
        self,
        customer_id: str,
        shap_explainer: Any,
        customer_idx: int = 0,
    ) -> bytes:
        """Render SHAP waterfall plot and return as PNG bytes.

        Uses the :class:`~src.models.explainability.shap_explainer.ChurnSHAPExplainer`
        to generate a waterfall plot for a single customer, then encodes to PNG
        bytes for display via ``st.image``.

        Args:
            customer_id: Customer identifier (used in temp file naming).
            shap_explainer: An instance of ``ChurnSHAPExplainer`` with
                ``shap_values`` and ``base_value`` attributes.
            customer_idx: Row index in the test set to explain.

        Returns:
            PNG image bytes. Returns an empty bytes object if SHAP or
            matplotlib is not available.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap

            shap_vals = shap_explainer.shap_values
            if shap_vals is None or len(shap_vals) == 0:
                return b""

            buf = io.BytesIO()
            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[customer_idx],
                    base_values=shap_explainer.base_value,
                    feature_names=shap_explainer.feature_names,
                ),
                show=False,
                max_display=12,
            )
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return buf.read()
        except ImportError:
            logger.warning("shap or matplotlib not installed; returning empty bytes.")
            return b""
        except Exception as exc:
            logger.error("Failed to build SHAP waterfall for customer %s: %s", customer_id, exc)
            return b""

    # ------------------------------------------------------------------
    # Retention action formatting
    # ------------------------------------------------------------------

    def format_retention_actions(
        self,
        shap_row: dict[str, float],
        customer_features: dict[str, Any] | None = None,
        max_actions: int = 3,
    ) -> list[dict[str, str]]:
        """Generate structured retention action recommendations from SHAP values.

        Rule-based mapping:
        - High ``recency_days`` SHAP → re-engagement email.
        - High ``frequency`` SHAP → loyalty reward push notification.
        - High ``monetary`` SHAP → high-value product recommendation.

        Args:
            shap_row: Dict of ``{feature_name: shap_value}`` for one customer.
            customer_features: Optional raw feature dict for personalising
                action text (e.g., inserting actual recency_days value).
            max_actions: Maximum number of actions to return.

        Returns:
            List of up to ``max_actions`` dicts, each containing:
            ``action``, ``urgency``, ``channel``, ``expected_lift_pct``.
        """
        recency_days = (customer_features or {}).get("recency_days", 45)
        actions: list[dict[str, str]] = []

        # Rule 1: Recency driver
        if shap_row.get("recency_days", 0.0) > 0.05:
            urgency = "High" if recency_days > 60 else "Medium"
            actions.append({
                "action": (
                    f"Re-engage via personalised email — last purchase was {recency_days} days ago. "
                    "Send targeted win-back offer with exclusive discount."
                ),
                "urgency": urgency,
                "channel": "Email",
                "expected_lift_pct": "12–18%",
            })

        # Rule 2: Frequency driver
        if shap_row.get("frequency", 0.0) > -0.02:
            actions.append({
                "action": (
                    "Offer loyalty reward — purchase frequency has declined. "
                    "Award double loyalty points on next qualifying purchase."
                ),
                "urgency": "Medium",
                "channel": "Push Notification",
                "expected_lift_pct": "8–14%",
            })

        # Rule 3: Monetary driver
        if shap_row.get("monetary", 0.0) > 0.03:
            actions.append({
                "action": (
                    "Personalised high-value product recommendation based on past purchase affinity "
                    "and browsing history. Upsell via targeted in-app card."
                ),
                "urgency": "Low",
                "channel": "In-App",
                "expected_lift_pct": "6–10%",
            })

        # Rule 4: Rolling mean driver
        if shap_row.get("rolling_mean_7d", 0.0) > 0.05:
            actions.append({
                "action": (
                    "Weekly purchase pattern disrupted — send SMS reminder with cart abandonment incentive."
                ),
                "urgency": "High",
                "channel": "SMS",
                "expected_lift_pct": "10–16%",
            })

        # Fallback action
        if not actions:
            actions.append({
                "action": "Send re-activation SMS with exclusive 15% discount code for next 7 days.",
                "urgency": "High",
                "channel": "SMS",
                "expected_lift_pct": "10–16%",
            })

        return actions[:max_actions]

    # ------------------------------------------------------------------
    # Segment badge HTML
    # ------------------------------------------------------------------

    def get_segment_badge_html(
        self,
        persona: str,
        extra_style: str = "",
    ) -> str:
        """Return inline HTML badge for a customer segment persona.

        Args:
            persona: Persona label string (e.g., "Champion", "At-Risk VIP").
            extra_style: Optional additional inline CSS to append.

        Returns:
            HTML ``<span>`` string for ``st.markdown(unsafe_allow_html=True)``.
        """
        bg, text_color = _PERSONA_BADGE_MAP.get(persona, ("#F3F4F6", "#374151"))
        return (
            f'<span style="background:{bg}; color:{text_color}; '
            f"padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; "
            f'font-weight: 600; {extra_style}">{persona}</span>'
        )

    # ------------------------------------------------------------------
    # Customer risk tier badge
    # ------------------------------------------------------------------

    def get_risk_tier_html(self, churn_proba: float) -> str:
        """Return inline HTML badge for a customer's churn risk tier.

        Args:
            churn_proba: Churn probability in [0, 1].

        Returns:
            HTML ``<span>`` string for ``st.markdown(unsafe_allow_html=True)``.
        """
        if churn_proba > 0.80:
            label, bg, color = "Critical", "#dc2626", "white"
        elif churn_proba > 0.60:
            label, bg, color = "High", PRIMARY, "white"
        elif churn_proba > 0.40:
            label, bg, color = "Medium", SECONDARY, "white"
        else:
            label, bg, color = "Low", "#16a34a", "white"
        return (
            f'<span style="background:{bg}; color:{color}; '
            f"padding: 4px 12px; border-radius: 6px; font-size: 0.82rem; "
            f'font-weight: 700;">{label} Risk ({churn_proba:.1%})</span>'
        )
