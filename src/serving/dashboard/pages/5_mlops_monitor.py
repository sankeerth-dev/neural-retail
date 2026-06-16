"""Page 5 — MLOps Monitor Dashboard.

NeuralRetail Intelligence Platform · AMX-DS-2026-04
Model performance trends, Evidently-style PSI drift bars, model registry,
retrain history, and manual retrain trigger UI.
Admin-only page. All data synthesised in-process.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    if st.session_state.get("role") != "Admin":
        st.error("🔒 This page is restricted to **Admin** users only.")
        st.info("Contact your platform administrator to request Admin access.")
        return False
    return True


# ── Data generators ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _perf_trend(days: int = 30) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    mape  = np.clip(9.2 + rng.normal(0, 0.4, days).cumsum() * 0.05, 6, 14)
    auc   = np.clip(0.912 + rng.normal(0, 0.003, days).cumsum() * 0.001, 0.85, 0.96)
    rmse  = 12.5 + rng.normal(0, 0.8, days)
    return pd.DataFrame({"date": dates, "demand_mape": mape, "churn_auc": auc, "demand_rmse": rmse})


@st.cache_data(ttl=300)
def _psi_data() -> pd.DataFrame:
    features = ["recency_days","frequency","monetary","rolling_mean_7d","lag_1d",
                "temp_c","cpi_index","is_promotional_period","day_of_week","days_to_next_holiday"]
    rng = np.random.default_rng(77)
    psi = rng.uniform(0.02, 0.38, len(features))
    ks  = rng.uniform(0.02, 0.18, len(features))
    return pd.DataFrame({"feature": features, "psi": psi.round(4), "ks_stat": ks.round(4),
                         "drift": psi > 0.20}).sort_values("psi", ascending=False)


@st.cache_data(ttl=300)
def _model_registry() -> pd.DataFrame:
    return pd.DataFrame({
        "model_name":     ["demand_ensemble","demand_ensemble","churn_stacking_ensemble",
                           "churn_stacking_ensemble","kmeans_segmentation",
                           "price_elasticity_electronics","lstm_forecaster_A"],
        "version":        [3, 2, 5, 4, 2, 1, 1],
        "stage":          ["Production","Archived","Production","Staging","Production","Production","Staging"],
        "deployed_at":    ["2026-05-25 09:15","2026-05-18 14:30","2026-05-24 11:00",
                           "2026-05-27 16:45","2026-05-22 08:00","2026-05-20 10:30","2026-05-27 20:00"],
        "primary_metric": ["MAPE","MAPE","AUC-ROC","AUC-ROC","Silhouette","R²","MAPE"],
        "metric_value":   [0.087, 0.094, 0.921, 0.908, 0.572, 0.741, 0.093],
        "status":         ["✅ Healthy","📦 Archived","✅ Healthy","🧪 Challenger",
                           "✅ Healthy","✅ Healthy","🔬 Testing"],
    })


@st.cache_data(ttl=300)
def _retrain_history() -> pd.DataFrame:
    return pd.DataFrame({
        "trigger_time":    ["2026-05-25 02:00","2026-05-22 02:00","2026-05-18 02:00",
                            "2026-05-14 02:00","2026-05-07 02:00"],
        "trigger_reason":  ["PSI=0.24 (recency_days)","Scheduled weekly","MAPE degradation +15%",
                            "PSI=0.21 (monetary)","Scheduled weekly"],
        "models_retrained":["demand_ensemble, churn_stacking","All models","demand_ensemble",
                            "churn_stacking","All models"],
        "duration_min":    [17.4, 19.1, 11.2, 14.8, 18.5],
        "outcome":         ["✅ Pass","✅ Pass","✅ Pass","⚠️ Partial","✅ Pass"],
        "mape_before":     [10.3, 9.8, 11.7, 9.5, 9.2],
        "mape_after":      [8.7, 9.1, 9.4, 9.0, 8.9],
    })


# ── Charts ────────────────────────────────────────────────────────────────
def _perf_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["demand_mape"],
                             name="Demand MAPE (%)", line=dict(color=PRIMARY, width=2.5), yaxis="y"))
    fig.add_hline(y=10.0, line_dash="dash", line_color=PRIMARY, line_width=1.4,
                  annotation_text="MAPE target (10%)", annotation_font_color="#111827", yref="y")
    fig.add_hrect(y0=10.0, y1=14.5, fillcolor=f"rgba(232,78,27,.05)", line_width=0, yref="y")
    fig.add_trace(go.Scatter(x=df["date"], y=df["churn_auc"],
                             name="Churn AUC-ROC", line=dict(color=SECONDARY, width=2.5, dash="dot"), yaxis="y2"))
    fig.add_hline(y=0.90, line_dash="dot", line_color=SECONDARY, line_width=1.4,
                  annotation_text="AUC target (0.90)", annotation_font_color="#111827", yref="y2")
    fig.update_layout(
        title="Model Performance — Last 30 Days",
        xaxis=dict(gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(title="MAPE (%)", gridcolor="#f5f5f5", range=[5,15],
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis2=dict(title="AUC-ROC", overlaying="y", side="right", range=[.84,.97],
                    tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#111827"),
        legend=dict(orientation="h", y=-0.2, font=dict(color="#111827")),
        title_font=dict(color="#111827"),
        margin=dict(l=50, r=50, t=50, b=70),
    )
    return fig


def _psi_bar_fig(df: pd.DataFrame) -> go.Figure:
    colors = [PRIMARY if p > 0.2 else SECONDARY if p > 0.1 else "#16a34a" for p in df["psi"]]
    fig = go.Figure(go.Bar(
        x=df["psi"], y=df["feature"], orientation="h",
        marker=dict(color=colors),
        text=[f"PSI: {p:.3f}" + (" ⚠️" if p > 0.2 else "") for p in df["psi"]],
        textposition="outside",
        hovertemplate="%{y}: PSI=%{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color=SECONDARY, annotation_text="Moderate (0.1)",
                  annotation_font_color="#111827")
    fig.add_vline(x=0.2, line_dash="dash", line_color=PRIMARY,   annotation_text="Severe (0.2)",
                  annotation_font_color="#111827")
    fig.update_layout(
        title="Data Drift PSI — Top SHAP Features (🟢<0.1 | 🟡 0.1-0.2 | 🔴>0.2)",
        xaxis=dict(title="PSI Score", gridcolor="#f5f5f5", range=[0, 0.45],
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(gridcolor="#f5f5f5", tickfont=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#111827"),
        title_font=dict(color="#111827"),
        margin=dict(l=20, r=100, t=55, b=50), height=380,
    )
    return fig


def _retrain_timeline_fig(df: pd.DataFrame) -> go.Figure:
    outcome_colors = {"✅ Pass":"#16a34a","⚠️ Partial":ACCENT,"❌ Fail":PRIMARY}
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["trigger_time"]], y=[row["duration_min"]],
            mode="markers+text",
            marker=dict(size=24, color=outcome_colors.get(row["outcome"],"#888"),
                        line=dict(color="white", width=2)),
            text=[row["outcome"]], textposition="top center",
            name=row["trigger_reason"][:30],
            hovertemplate=(
                f"<b>{row['trigger_time']}</b><br>"
                f"Reason: {row['trigger_reason']}<br>"
                f"MAPE: {row['mape_before']}% → {row['mape_after']}%<extra></extra>"
            ),
        ))
    fig.add_hline(y=20, line_dash="dash", line_color=PRIMARY, annotation_text="SLA 20 min",
                  annotation_font_color="#111827")
    fig.update_layout(
        title="Retrain Pipeline History",
        xaxis=dict(title="Trigger Time", type="category", gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(title="Duration (min)", range=[0, 25], gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#111827"),
        title_font=dict(color="#111827"),
        showlegend=False, margin=dict(l=50, r=20, t=55, b=80), height=320,
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────
def render() -> None:
    _css()
    if not _check_auth():
        return

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">🔬 MLOps Monitor</div>'
        '<div class="page-subtitle">Model performance trends, data drift PSI, model registry, retrain history, and pipeline controls</div>',
        unsafe_allow_html=True,
    )

    perf_df     = _perf_trend()
    psi_df      = _psi_data()
    registry_df = _model_registry()
    retrain_df  = _retrain_history()

    # Header KPIs
    latest = perf_df.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current MAPE",     f"{latest['demand_mape']:.2f}%",
              delta=f"{latest['demand_mape'] - 10:.2f}pp vs target", delta_color="inverse")
    k2.metric("Current AUC",      f"{latest['churn_auc']:.4f}",
              delta=f"+{latest['churn_auc'] - 0.90:.4f} vs target")
    k3.metric("Drifted Features", f"{psi_df['drift'].sum()}/{len(psi_df)}",
              delta="PSI > 0.20")
    k4.metric("Production Models", f"{(registry_df['stage']=='Production').sum()}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Performance Trends","🌊 Drift Monitor","📋 Model Registry","⏳ Retrain History","🚀 Pipeline Controls"]
    )

    with tab1:
        st.markdown('<div class="section-header">Model Performance — Last 30 Days</div>', unsafe_allow_html=True)
        st.plotly_chart(_perf_fig(perf_df), use_container_width=True, key="ml_perf")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Avg MAPE (30d)",  f"{perf_df['demand_mape'].mean():.2f}%")
        tc2.metric("Min MAPE (best)", f"{perf_df['demand_mape'].min():.2f}%")
        tc3.metric("RMSE (current)",  f"{latest['demand_rmse']:.2f} units")

    with tab2:
        st.markdown('<div class="section-header">Data Drift PSI Dashboard</div>', unsafe_allow_html=True)
        st.plotly_chart(_psi_bar_fig(psi_df), use_container_width=True, key="ml_psi")
        d1, d2, d3 = st.columns(3)
        d1.metric("Features with Drift", f"{psi_df['drift'].sum()}/{len(psi_df)}")
        d2.metric("Max PSI",             f"{psi_df['psi'].max():.3f} ({psi_df.iloc[0]['feature']})")
        d3.metric("Drift Status",        "⚠️ DRIFT DETECTED" if psi_df["drift"].any() else "✅ Stable")

        st.markdown("<br>**Drift Detail Table**")
        disp = psi_df.copy()
        disp["drift"] = disp["drift"].map({True:"⚠️ YES", False:"✅ No"})
        st.dataframe(disp, use_container_width=True,
                     column_config={
                         "psi":     st.column_config.NumberColumn("PSI Score",   format="%.4f"),
                         "ks_stat": st.column_config.NumberColumn("KS Stat",     format="%.4f"),
                         "drift":   st.column_config.TextColumn("Drift"),
                     })

    with tab3:
        st.markdown('<div class="section-header">Active Model Registry</div>', unsafe_allow_html=True)

        def _stage_style(val: str) -> str:
            if val == "Production": return "background-color:#E1F5EE;color:#085041;font-weight:600"
            if val == "Staging":    return "background-color:#FEF3C7;color:#92400E;font-weight:600"
            return ""

        styled_reg = registry_df.style.map(_stage_style, subset=["stage"])
        st.dataframe(styled_reg, use_container_width=True,
                     column_config={
                         "metric_value": st.column_config.NumberColumn("Metric Value", format="%.4f"),
                         "version":      st.column_config.NumberColumn("Version",      format="%d"),
                     })
        reg_c1, reg_c2 = st.columns(2)
        reg_c1.metric("Production Models", (registry_df["stage"] == "Production").sum())
        reg_c2.metric("Challenger Models", (registry_df["stage"] == "Staging").sum())

    with tab4:
        st.markdown('<div class="section-header">Retrain Pipeline History</div>', unsafe_allow_html=True)
        st.plotly_chart(_retrain_timeline_fig(retrain_df), use_container_width=True, key="ml_retrain")
        st.dataframe(retrain_df, use_container_width=True,
                     column_config={
                         "duration_min":  st.column_config.NumberColumn("Duration (min)", format="%.1f"),
                         "mape_before":   st.column_config.NumberColumn("MAPE Before (%)", format="%.1f"),
                         "mape_after":    st.column_config.NumberColumn("MAPE After (%)", format="%.1f"),
                     })

    with tab5:
        st.markdown('<div class="section-header">🚀 Pipeline Controls (Admin)</div>', unsafe_allow_html=True)
        st.warning("⚠️ **Admin action** — Manual retrain consumes significant compute. Confirm before proceeding.")

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Manual Retrain Trigger**")
            reason = st.text_input("Retrain Reason (audit log)",
                                   placeholder="e.g., Manual trigger — PSI exceeded 0.20",
                                   key="ml_reason")
            models = st.multiselect("Models to Retrain",
                                    ["demand_ensemble","churn_stacking","kmeans_segmentation","price_elasticity"],
                                    default=["demand_ensemble"], key="ml_models")
            confirmed = st.checkbox("✅ I confirm this action", key="ml_confirm")

            if st.button("🚀 Trigger Retrain Now", type="primary",
                         disabled=not confirmed, key="ml_retrain_btn"):
                if not models:
                    st.error("Select at least one model to retrain.")
                else:
                    with st.spinner("Submitting retrain job to Airflow…"):
                        import time; time.sleep(1.5)
                    st.success(
                        f"✅ Retrain pipeline triggered for: **{', '.join(models)}**\n\n"
                        f"Reason: _{reason or 'Manual trigger'}_\n\n"
                        "Expected duration: **15–20 minutes** | Monitor in Airflow UI."
                    )
                    st.info("🔗 **Airflow URL:** `http://localhost:8080` → DAG: `neuralretail_retrain_pipeline`")

        with pc2:
            st.markdown("**System Info**")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">MLflow Tracking</div>
                    <div style="font-size:.9rem;">http://localhost:5000</div>
                    <div class="metric-label" style="margin-top:.7rem;">Retrain DAG</div>
                    <div style="font-size:.9rem;">neuralretail_retrain_pipeline</div>
                    <div class="metric-label" style="margin-top:.7rem;">Schedule</div>
                    <div style="font-size:.9rem;">Trigger-only (no cron)</div>
                    <div class="metric-label" style="margin-top:.7rem;">SLA</div>
                    <div style="font-size:.9rem;">≤ 20 minutes end-to-end</div>
                    <div class="metric-label" style="margin-top:.7rem;">Last Retrain</div>
                    <div style="font-size:.9rem;">2026-05-25 02:00 UTC — ✅ Pass (17.4 min)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>")
        st.markdown('<div class="section-header">Gate Thresholds</div>', unsafe_allow_html=True)
        th_cols = st.columns(4)
        thresholds = [
            ("Demand MAPE Gate",     "≤ 10.0%",  "✅ Pass (8.7%)"),
            ("Churn AUC Gate",       "≥ 0.900",  "✅ Pass (0.921)"),
            ("PSI Drift Threshold",  "> 0.200",  "⚠️ 2 features above"),
            ("MAPE Degradation SLA", "> 15%",    "✅ No degradation"),
        ]
        for col, (lbl, thr, status) in zip(th_cols, thresholds):
            is_pass = "✅" in status
            col.markdown(
                f'<div class="metric-card" style="text-align:center;padding:1rem;">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div style="font-size:1rem;font-weight:700;color:#1a1a1a;">{thr}</div>'
                f'<span class="{"status-badge-pass" if is_pass else "status-badge-warn"}">{status}</span></div>',
                unsafe_allow_html=True,
            )


render()
