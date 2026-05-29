"""Page 1 — Executive KPI Dashboard.

NeuralRetail Intelligence Platform · AMX-DS-2026-04
All data is synthesised in-process — no external dependencies.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Brand ────────────────────────────────────────────────────────────────
PRIMARY   = "#E84E1B"
SECONDARY = "#F7941D"
ACCENT    = "#FBBA13"
_CSS = Path(__file__).parents[1] / "assets" / "style.css"


def _css() -> None:
    if _CSS.exists():
        st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)


def _check_auth() -> bool:
    if not st.session_state.get("logged_in", False):
        st.warning("🔒 Please sign in from the Home page.")
        st.page_link("app.py", label="← Go to Home / Sign In")
        return False
    return True


# ── Mock data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _revenue_trend(days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range(end=date.today(), periods=days, freq="D")
    base  = np.linspace(80_000, 110_000, days)
    noise = rng.normal(0, 4000, days)
    promo = np.where((np.arange(days) % 14) == 0, 18_000, 0)
    return pd.DataFrame({"date": dates, "revenue": (base + noise + promo).clip(0)})


@st.cache_data(ttl=300)
def _category_mix() -> pd.DataFrame:
    return pd.DataFrame({
        "category": ["Electronics", "Apparel", "Food & Bev", "Health", "Home & Garden"],
        "revenue":  [1_240_000, 820_000, 540_000, 380_000, 270_000],
        "margin":   [28.4, 42.1, 18.7, 35.2, 31.0],
    })


@st.cache_data(ttl=300)
def _mape_trend(days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=date.today(), periods=days, freq="D")
    mape  = np.clip(9.5 + rng.normal(0, 0.4, days).cumsum() * 0.05, 6, 13)
    auc   = np.clip(0.912 + rng.normal(0, 0.003, days).cumsum() * 0.001, 0.86, 0.96)
    return pd.DataFrame({"date": dates, "demand_mape": mape, "churn_auc": auc})


# ── Chart builders ────────────────────────────────────────────────────────
def _revenue_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["revenue"],
        fill="tozeroy", fillcolor=f"rgba(232,78,27,.10)",
        line=dict(color=PRIMARY, width=2.5),
        name="Daily Revenue",
    ))
    # 7-day MA
    df = df.copy()
    df["ma7"] = df["revenue"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ma7"],
        line=dict(color=SECONDARY, width=1.8, dash="dot"),
        name="7-day MA",
    ))
    fig.update_layout(
        title="Daily Revenue (£)", hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=45, b=40),
        xaxis=dict(gridcolor="#f5f5f5", zeroline=False),
        yaxis=dict(gridcolor="#f5f5f5", zeroline=False, tickprefix="£", tickformat=",.0f"),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def _category_bar(df: pd.DataFrame) -> go.Figure:
    colors = [PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7"]
    fig = go.Figure(go.Bar(
        x=df["category"], y=df["revenue"],
        marker=dict(color=colors),
        text=[f"£{v/1e6:.2f}M" for v in df["revenue"]],
        textposition="outside",
        hovertemplate="%{x}<br>Revenue: £%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Revenue by Category", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        yaxis=dict(gridcolor="#f5f5f5", tickprefix="£", tickformat=",.0f"),
        margin=dict(l=40, r=20, t=45, b=40),
    )
    return fig


def _gauge_fig(value: float, threshold: float, title: str, suffix: str = "%",
               invert: bool = False) -> go.Figure:
    pct = (value / threshold) * 100 if threshold else 0
    good = pct <= 100 if invert else pct >= 100
    color = "#16a34a" if good else PRIMARY
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number=dict(suffix=suffix, font=dict(size=28, color=color)),
        delta=dict(reference=threshold, suffix=suffix,
                   increasing=dict(color="#16a34a" if not invert else PRIMARY),
                   decreasing=dict(color=PRIMARY if not invert else "#16a34a")),
        gauge=dict(
            axis=dict(range=[0, threshold * 1.4]),
            bar=dict(color=color),
            bgcolor="white",
            borderwidth=0,
            steps=[dict(range=[0, threshold], color="#f5f5f5")],
            threshold=dict(line=dict(color=PRIMARY, width=3), thickness=0.78, value=threshold),
        ),
        title=dict(text=title, font=dict(size=13)),
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="white")
    return fig


def _mape_auc_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["demand_mape"],
        name="Demand MAPE (%)", line=dict(color=PRIMARY, width=2.5), yaxis="y",
    ))
    fig.add_hline(y=10, line_dash="dash", line_color=PRIMARY, line_width=1.2,
                  annotation_text="MAPE target 10%", yref="y")
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["churn_auc"],
        name="Churn AUC-ROC", line=dict(color=SECONDARY, width=2.5, dash="dot"), yaxis="y2",
    ))
    fig.add_hline(y=0.90, line_dash="dot", line_color=SECONDARY, line_width=1.2,
                  annotation_text="AUC target 0.90", yref="y2")
    fig.update_layout(
        title="Model Performance — Last 30 Days",
        xaxis=dict(gridcolor="#f5f5f5", zeroline=False),
        yaxis=dict(title="MAPE (%)", gridcolor="#f5f5f5", range=[5, 15]),
        yaxis2=dict(title="AUC-ROC", overlaying="y", side="right", range=[0.84, 0.97],
                    gridcolor="rgba(0,0,0,0)"),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=50, r=50, t=45, b=70),
    )
    return fig


# ── Export helpers ────────────────────────────────────────────────────────
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Page ──────────────────────────────────────────────────────────────────
def render() -> None:
    _css()
    if not _check_auth():
        return

    role = st.session_state.get("role", "Viewer")

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">📊 Executive KPI Dashboard</div>'
        '<div class="page-subtitle">Board-ready performance summary — refreshed every 5 minutes</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar filters ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        days = st.slider("Revenue history (days)", 30, 180, 90, key="kpi_days")
        store_filter = st.selectbox("Store", ["All Stores","London Central","Manchester","Birmingham","Edinburgh"], key="kpi_store")

    rev_df  = _revenue_trend(days)
    cat_df  = _category_mix()
    perf_df = _mape_trend()

    # ── Top KPI row ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "£2.84M",  "Total Revenue",      "+12.3%",  True),
        (k2, "8.7%",    "Demand MAPE",         "↓1.3pp",  True),
        (k3, "0.921",   "Churn AUC",           "+0.011",  True),
        (k4, "4.2%",    "Stockout Rate",        "▼0.8pp",  True),
        (k5, "0.572",   "Seg Silhouette",       "+0.017",  True),
        (k6, "0.741",   "Price Elasticity R²",  "+0.029",  True),
    ]
    for col, val, label, delta, pos in kpis:
        dcls = "metric-delta-positive" if pos else "metric-delta-negative"
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="font-size:1.6rem;">{val}</div>'
            f'<div class="{dcls}">{delta}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue + Category ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Revenue Trend", "🗂️ Category Mix", "🎯 Model Health", "⬇️ Exports"]
    )

    with tab1:
        st.markdown('<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(_revenue_fig(rev_df), use_container_width=True, key="exec_rev")

        total = rev_df["revenue"].sum()
        avg   = rev_df["revenue"].mean()
        peak  = rev_df["revenue"].max()
        c1, c2, c3 = st.columns(3)
        c1.metric("Period Revenue", f"£{total:,.0f}")
        c2.metric("Daily Average",  f"£{avg:,.0f}")
        c3.metric("Peak Day",       f"£{peak:,.0f}")

    with tab2:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        st.plotly_chart(_category_bar(cat_df), use_container_width=True, key="exec_cat")

        st.dataframe(
            cat_df.rename(columns={"category":"Category","revenue":"Revenue (£)","margin":"Gross Margin (%)"}),
            use_container_width=True,
            column_config={
                "Revenue (£)":       st.column_config.NumberColumn(format="£%,.0f"),
                "Gross Margin (%)":  st.column_config.ProgressColumn(min_value=0, max_value=60),
            },
        )

    with tab3:
        st.markdown('<div class="section-header">Model Performance Gauges</div>', unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        g1.plotly_chart(_gauge_fig(8.7,  10.0, "Demand MAPE",   "%",  invert=True),  use_container_width=True, key="g_mape")
        g2.plotly_chart(_gauge_fig(0.921, 0.90, "Churn AUC",    "",   invert=False), use_container_width=True, key="g_auc")
        g3.plotly_chart(_gauge_fig(0.572, 0.55, "Seg Sil.",     "",   invert=False), use_container_width=True, key="g_sil")
        g4.plotly_chart(_gauge_fig(0.741, 0.72, "Price R²",     "",   invert=False), use_container_width=True, key="g_r2")

        st.markdown('<div class="section-header">30-Day Performance Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(_mape_auc_fig(perf_df), use_container_width=True, key="exec_perf")

    with tab4:
        st.markdown('<div class="section-header">Data Exports</div>', unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**Revenue Data**")
            st.download_button(
                "⬇️ Download Revenue CSV",
                _csv_bytes(rev_df),
                f"NeuralRetail_Revenue_{date.today()}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_revenue",
            )
        with ec2:
            st.markdown("**Category Mix**")
            st.download_button(
                "⬇️ Download Category CSV",
                _csv_bytes(cat_df),
                f"NeuralRetail_Categories_{date.today()}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_cat",
            )
        if role == "Admin":
            st.markdown("**Full Model Performance Log**")
            st.download_button(
                "⬇️ Download Model KPI CSV",
                _csv_bytes(perf_df),
                f"NeuralRetail_ModelKPI_{date.today()}.csv",
                "text/csv",
                use_container_width=True,
                key="dl_perf",
            )


render()
