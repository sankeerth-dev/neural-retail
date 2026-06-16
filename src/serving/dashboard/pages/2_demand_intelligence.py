"""Page 2 — Demand Intelligence Dashboard.

NeuralRetail Intelligence Platform · AMX-DS-2026-04
SKU demand forecasts, MAPE leaderboard, seasonal decomposition,
and what-if scenario simulator. All data synthesised in-process.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

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


# ── Data generators ───────────────────────────────────────────────────────
_SKUS = [f"SKU-{1000+i}" for i in range(20)]
_CATS = ["Electronics","Apparel","Food & Bev","Health","Home"]

@st.cache_data(ttl=300)
def _forecast_df(sku_id: str, horizon: int = 30, history: int = 90) -> pd.DataFrame:
    seed = hash(sku_id) % 2**31
    rng  = np.random.default_rng(seed)
    base = rng.uniform(100, 400)
    trend = np.linspace(0, rng.uniform(-20, 40), history + horizon)
    noise_std = base * 0.10

    dates_hist = pd.date_range(end=date.today(), periods=history, freq="D")
    dates_fore = pd.date_range(start=date.today() + timedelta(1), periods=horizon, freq="D")
    all_dates  = pd.concat([pd.Series(dates_hist), pd.Series(dates_fore)]).reset_index(drop=True)

    hist_vals = (base + trend[:history] + rng.normal(0, noise_std, history)).clip(0)
    p50_fore  = (base + trend[history:] + rng.normal(0, noise_std * 0.3, horizon)).clip(0)
    p10_fore  = (p50_fore - rng.uniform(noise_std * 0.5, noise_std * 1.2, horizon)).clip(0)
    p90_fore  = (p50_fore + rng.uniform(noise_std * 0.6, noise_std * 1.6, horizon)).clip(0)

    actual = np.concatenate([hist_vals, np.full(horizon, np.nan)])
    p10    = np.concatenate([np.full(history, np.nan), p10_fore])
    p50    = np.concatenate([np.full(history, np.nan), p50_fore])
    p90    = np.concatenate([np.full(history, np.nan), p90_fore])

    return pd.DataFrame({"date": all_dates, "actual": actual, "p10": p10, "p50": p50, "p90": p90})


@st.cache_data(ttl=300)
def _mape_leaderboard() -> pd.DataFrame:
    rng = np.random.default_rng(55)
    cats = rng.choice(_CATS, len(_SKUS))
    mape = rng.uniform(5, 18, len(_SKUS)).round(2)
    return pd.DataFrame({"sku_id": _SKUS, "category": cats, "mape": mape,
                         "vs_target": (mape - 10.0).round(2)}).sort_values("mape")


@st.cache_data(ttl=300)
def _seasonal_df(sku_id: str, weeks: int = 52) -> pd.DataFrame:
    seed = hash(sku_id) % 2**31 + 1
    rng  = np.random.default_rng(seed)
    dates = pd.date_range(end=date.today(), periods=weeks * 7, freq="D")
    day   = np.arange(len(dates))
    trend = 200 + day * 0.15
    weekly = 20 * np.sin(2 * math.pi * day / 7)
    annual = 35 * np.sin(2 * math.pi * day / 365 - 0.5)
    noise  = rng.normal(0, 12, len(dates))
    val    = (trend + weekly + annual + noise).clip(0)
    return pd.DataFrame({"date": dates, "observed": val,
                         "trend": trend, "seasonal": weekly + annual, "residual": noise})


# ── Chart builders ────────────────────────────────────────────────────────
def _forecast_fig(df: pd.DataFrame, sku: str) -> go.Figure:
    fig = go.Figure()
    hist = df[df["actual"].notna()]
    fore = df[df["p50"].notna()]

    # CI band
    x_band = list(fore["date"]) + list(fore["date"])[::-1]
    y_band = list(fore["p90"]) + list(fore["p10"])[::-1]
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself",
                             fillcolor="rgba(251,186,19,.22)",
                             line=dict(color="rgba(0,0,0,0)"),
                             name="P10–P90 CI", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual"],
                             name="Actual", line=dict(color=PRIMARY, width=2.5)))
    fig.add_trace(go.Scatter(x=fore["date"], y=fore["p50"],
                             name="Forecast P50", line=dict(color=SECONDARY, width=2.5, dash="dash")))

    if not fore.empty:
        x_ms = int(fore["date"].min().timestamp() * 1000)
        fig.add_vline(x=x_ms, line_dash="dot", line_color="#aaa",
                      annotation_text="Forecast →", annotation_position="top right")

    fig.update_layout(
        title=f"Demand Forecast — {sku}",
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05), gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(gridcolor="#f5f5f5", title="Units",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#111827"),
        legend=dict(orientation="h", y=-0.22, font=dict(color="#111827")),
        title_font=dict(color="#111827"),
        margin=dict(l=50, r=20, t=50, b=70),
    )
    return fig


def _leaderboard_fig(df: pd.DataFrame) -> go.Figure:
    colors = [
        "#16a34a" if m <= 10 else SECONDARY if m <= 15 else PRIMARY
        for m in df["mape"]
    ]
    fig = go.Figure(go.Bar(
        x=df["mape"], y=df["sku_id"], orientation="h",
        marker=dict(color=colors),
        text=[f"{m:.1f}%" for m in df["mape"]], textposition="outside",
        hovertemplate="%{y} — MAPE: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=10, line_dash="dash", line_color=PRIMARY, annotation_text="Target 10%",
                  annotation_font_color="#111827")
    fig.add_vline(x=15, line_dash="dot",  line_color="#dc2626", annotation_text="Danger 15%",
                  annotation_font_color="#111827")
    fig.update_layout(
        title="MAPE Leaderboard (Green ≤10% | Amber ≤15% | Red >15%)",
        xaxis=dict(title="MAPE (%)", gridcolor="#f5f5f5", range=[0, 22],
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#111827"),
        title_font=dict(color="#111827"),
        margin=dict(l=20, r=80, t=50, b=40), height=520,
    )
    return fig


def _seasonal_fig(df: pd.DataFrame, sku: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["observed"],
                             name="Observed", line=dict(color=PRIMARY, width=1.8)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["trend"],
                             name="Trend", line=dict(color=SECONDARY, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["seasonal"] + df["observed"].mean(),
                             name="Seasonal Component", line=dict(color=ACCENT, width=1.5, dash="dash")))
    fig.update_layout(
        title=f"Seasonal Decomposition — {sku}",
        xaxis=dict(gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(gridcolor="#f5f5f5", title="Units",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#111827"),
        hovermode="x unified", legend=dict(orientation="h", y=-0.2, font=dict(color="#111827")),
        title_font=dict(color="#111827"),
        margin=dict(l=50, r=20, t=50, b=70),
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────
def render() -> None:
    _css()
    if not _check_auth():
        return

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">📈 Demand Intelligence</div>'
        '<div class="page-subtitle">SKU-level probabilistic forecasting with what-if scenario analysis</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Forecast Controls")
        sku = st.selectbox("Select SKU", _SKUS, key="di_sku")
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, key="di_horizon")
        history = st.slider("Historical Window (days)", 30, 180, 90, key="di_history")

    fc_df   = _forecast_df(sku, horizon, history)
    mape_df = _mape_leaderboard()
    seas_df = _seasonal_df(sku)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Forecast Chart", "🏆 MAPE Leaderboard", "🌊 Seasonal Decomposition", "⚙️ What-If Simulator"]
    )

    with tab1:
        st.markdown('<div class="section-header">Demand Forecast</div>', unsafe_allow_html=True)
        st.plotly_chart(_forecast_fig(fc_df, sku), use_container_width=True, key="di_fc")

        # Summary metrics
        fore_rows = fc_df[fc_df["p50"].notna()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total P50 Forecast", f"{fore_rows['p50'].sum():,.0f} units")
        c2.metric("Daily Avg P50",      f"{fore_rows['p50'].mean():,.1f} units")
        c3.metric("P90 Peak",           f"{fore_rows['p90'].max():,.0f} units")
        c4.metric("P10 Min",            f"{fore_rows['p10'].min():,.0f} units")

        st.download_button(
            "⬇️ Download Forecast CSV",
            fc_df.to_csv(index=False).encode("utf-8-sig"),
            f"NeuralRetail_Forecast_{sku}_{date.today()}.csv",
            "text/csv", key="dl_fc",
        )

    with tab2:
        st.markdown('<div class="section-header">SKU MAPE Leaderboard</div>', unsafe_allow_html=True)
        st.plotly_chart(_leaderboard_fig(mape_df), use_container_width=True, key="di_mape")

        above = (mape_df["mape"] > 10).sum()
        below = (mape_df["mape"] <= 10).sum()
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("SKUs Within Target (≤10%)", below)
        lc2.metric("SKUs Above Target (>10%)", above)
        lc3.metric("Average MAPE",             f"{mape_df['mape'].mean():.2f}%")

    with tab3:
        st.markdown('<div class="section-header">Seasonal Decomposition</div>', unsafe_allow_html=True)
        st.plotly_chart(_seasonal_fig(seas_df, sku), use_container_width=True, key="di_seas")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Trend Direction",  "↑ Positive" if seas_df["trend"].iloc[-1] > seas_df["trend"].iloc[0] else "↓ Negative")
        sc2.metric("Seasonal Amplitude", f"±{seas_df['seasonal'].std():.1f} units")
        sc3.metric("Residual Std Dev",   f"{seas_df['residual'].std():.1f} units")

    with tab4:
        st.markdown('<div class="section-header">📊 What-If Scenario Simulator</div>', unsafe_allow_html=True)
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            promo_lift_pct  = st.slider("Promotional Lift (%)", -20, 50, 15, key="wi_promo")
            price_change_pct = st.slider("Price Change (%)",    -30, 30,  0, key="wi_price")
            temp_delta       = st.slider("Temperature Δ (°C)",  -10, 15,  0, key="wi_temp")
        with sc_col2:
            elasticity = -1.5  # Price elasticity constant
            demand_adj  = promo_lift_pct / 100.0 + elasticity * (price_change_pct / 100.0) + temp_delta * 0.005
            base_p50 = fc_df[fc_df["p50"].notna()]["p50"].sum()
            adj_total = base_p50 * (1 + demand_adj)

            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;">
                    <div class="metric-label">Adjusted Total Forecast</div>
                    <div class="metric-value">{adj_total:,.0f} units</div>
                    <div class="{'metric-delta-positive' if adj_total > base_p50 else 'metric-delta-negative'}">
                        {'+' if adj_total > base_p50 else ''}{((adj_total/base_p50-1)*100):.1f}% vs base
                    </div>
                </div>
                <br>
                <div class="metric-card" style="text-align:center;">
                    <div class="metric-label">Base Forecast (P50)</div>
                    <div class="metric-value-secondary">{base_p50:,.0f} units</div>
                    <div class="tooltip-text">{horizon}-day horizon for {sku}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


render()
