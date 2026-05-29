"""Page 4 — Inventory Health Dashboard.

NeuralRetail Intelligence Platform · AMX-DS-2026-04
ABC-XYZ matrix, EOQ calculator, reorder alerts, overstock analysis,
and supplier lead-time analysis. All data synthesised in-process.
"""

from __future__ import annotations

import math
from datetime import date
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
    role = st.session_state.get("role", "Viewer")
    if role not in ("Admin", "Analyst"):
        st.warning("🔒 Analyst or Admin role required.")
        return False
    return True


# ── Data generators ───────────────────────────────────────────────────────
_SUPPLIERS = ["Supplier-A","Supplier-B","Supplier-C","Supplier-D","Supplier-E"]
_CATS      = ["Electronics","Apparel","Food & Bev","Health","Home & Garden"]


@st.cache_data(ttl=300)
def _abc_xyz_data() -> pd.DataFrame:
    rows = []
    for abc in ["A","B","C"]:
        for xyz in ["X","Y","Z"]:
            risk = {"A":.42,"B":.27,"C":.12}[abc]
            mod  = {"X":.08,"Y":.22,"Z":.48}[xyz]
            rng  = np.random.default_rng(hash(abc+xyz) % 2**31)
            rows.append({
                "abc_class": abc, "xyz_class": xyz,
                "stockout_risk": float(np.clip(rng.uniform(risk-.08, risk+mod+.12), 0, 1)),
                "sku_count": int(rng.integers(18, 110)),
                "avg_holding_cost": round(float(rng.uniform(600, 7000)), 2),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def _reorder_alerts(n: int = 35) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    days = rng.integers(1, 45, n)
    urgency = np.where(days <= 7, "Critical", np.where(days <= 14, "High", "Medium"))
    return pd.DataFrame({
        "sku_id":           [f"SKU-{2000+i}" for i in range(n)],
        "category":         rng.choice(_CATS, n),
        "current_stock":    rng.integers(10, 500, n),
        "reorder_point":    rng.integers(50, 300, n),
        "days_until_stockout": days,
        "urgency":          urgency,
        "supplier":         rng.choice(_SUPPLIERS, n),
    }).sort_values(["urgency","days_until_stockout"], ascending=[True, True])


@st.cache_data(ttl=300)
def _overstock_data(n: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(88)
    return pd.DataFrame({
        "sku_id":         [f"SKU-{3000+i}" for i in range(n)],
        "days_of_supply": rng.uniform(10, 260, n),
        "holding_cost":   rng.uniform(500, 35_000, n),
        "overstock_units":rng.integers(50, 2500, n),
        "dead_stock_risk":rng.uniform(0, 1, n),
        "category":       rng.choice(_CATS, n),
    })


@st.cache_data(ttl=300)
def _supplier_lt() -> pd.DataFrame:
    params = {"Supplier-A":(7,1.5),"Supplier-B":(14,4.0),"Supplier-C":(10,2.0),
              "Supplier-D":(21,8.0),"Supplier-E":(9,2.5)}
    rows = []
    for sup, (mean, std) in params.items():
        rng = np.random.default_rng(hash(sup) % 2**31)
        lts = np.clip(rng.normal(mean, std, 40), 1, None).round(0).astype(int)
        for lt in lts:
            rows.append({"supplier": sup, "lead_time_days": int(lt)})
    return pd.DataFrame(rows)


# ── EOQ ───────────────────────────────────────────────────────────────────
def _eoq(demand: float, order_cost: float, holding_pct: float, unit_cost: float,
         lead_time: int, z: float) -> dict:
    H = (holding_pct / 100.0) * unit_cost
    if H <= 0 or demand <= 0:
        return {k: 0.0 for k in ("eoq","ss","rop","tac","opy","acs")}
    eoq = math.sqrt(2 * demand * order_cost / H)
    daily = demand / 365.0
    ss    = z * math.sqrt(max(daily, 1)) * math.sqrt(lead_time)
    rop   = daily * lead_time + ss
    opy   = demand / eoq
    acs   = eoq / 2.0
    tac   = demand * unit_cost + opy * order_cost + (acs + ss) * H
    return {"eoq": round(eoq), "ss": round(ss), "rop": round(rop),
            "tac": round(tac, 2), "opy": round(opy, 1), "acs": round(acs)}


# ── Charts ────────────────────────────────────────────────────────────────
def _abc_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot     = df.pivot(index="abc_class", columns="xyz_class", values="stockout_risk")
    sku_pivot = df.pivot(index="abc_class", columns="xyz_class", values="sku_count")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#DCFCE7"],[0.4,ACCENT],[0.7,SECONDARY],[1.0,PRIMARY]],
        zmin=0, zmax=1,
        text=[[f"Risk: {pivot.values[r][c]:.1%}<br>SKUs: {sku_pivot.values[r][c]}"
               for c in range(3)] for r in range(3)],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="ABC: %{y}  XYZ: %{x}<br>Risk: %{z:.1%}<extra></extra>",
        colorbar=dict(title="Stockout Risk", tickformat=".0%"),
    ))
    fig.update_layout(title="ABC-XYZ Matrix — Stockout Risk",
                      xaxis=dict(title="Demand Variability (XYZ)"),
                      yaxis=dict(title="Revenue Contribution (ABC)"),
                      font=dict(family="Inter, sans-serif", size=12),
                      paper_bgcolor="white", margin=dict(l=60, r=20, t=60, b=60))
    return fig


def _overstock_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="days_of_supply", y="holding_cost", size="overstock_units",
                     color="dead_stock_risk",
                     color_continuous_scale=[[0,"#DCFCE7"],[.5,ACCENT],[1.0,PRIMARY]],
                     hover_data=["sku_id","category","overstock_units"],
                     labels={"days_of_supply":"Days of Supply",
                             "holding_cost":"Annual Holding Cost (£)",
                             "dead_stock_risk":"Dead Stock Risk"},
                     title="Overstock Risk Analysis")
    fig.add_vline(x=90,     line_dash="dash", line_color="#aaa", annotation_text="90-day threshold")
    fig.add_hline(y=10_000, line_dash="dash", line_color="#aaa", annotation_text="£10k holding cost")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, sans-serif", size=12),
                      margin=dict(l=50, r=20, t=60, b=50))
    return fig


def _supplier_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7"]
    for i, sup in enumerate(df["supplier"].unique()):
        sub = df[df["supplier"] == sup]["lead_time_days"]
        cv  = sub.std() / sub.mean() if sub.mean() > 0 else 0
        flag = cv > 0.3
        fig.add_trace(go.Box(y=sub, name=f"{'⚠️ ' if flag else ''}{sup}",
                             marker_color=PRIMARY if flag else colors[i % len(colors)],
                             boxmean="sd", line=dict(width=2)))
    fig.update_layout(title="Supplier Lead-Time Distribution (⚠️ = High CV)",
                      yaxis=dict(title="Lead Time (Days)", gridcolor="#f5f5f5"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, sans-serif", size=12),
                      showlegend=False, margin=dict(l=50, r=20, t=60, b=60))
    return fig


# ── Page ──────────────────────────────────────────────────────────────────
def render() -> None:
    _css()
    if not _check_auth():
        return

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">📦 Inventory Health</div>'
        '<div class="page-subtitle">ABC-XYZ matrix, EOQ calculator, reorder alerts, overstock analysis, and supplier intelligence</div>',
        unsafe_allow_html=True,
    )

    abc_df      = _abc_xyz_data()
    reorder_df  = _reorder_alerts()
    overstock_df= _overstock_data()
    supplier_df = _supplier_lt()

    # Header KPIs
    h1, h2, h3, h4 = st.columns(4)
    crit = (reorder_df["urgency"] == "Critical").sum()
    over90 = (overstock_df["days_of_supply"] > 90).sum()
    dead   = (overstock_df["dead_stock_risk"] > 0.7).sum()
    h1.metric("Critical Reorder Alerts", crit,  delta="≤ 7 days")
    h2.metric("SKUs > 90-Day Supply",  over90, delta="Overstock")
    h3.metric("Dead Stock Candidates", dead,   delta="Risk > 70%")
    h4.metric("Total SKUs Monitored",  abc_df["sku_count"].sum())

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔲 ABC-XYZ Matrix", "🧮 EOQ Calculator", "🚨 Reorder Alerts",
         "📊 Overstock Risk", "🚚 Supplier Lead Time"]
    )

    with tab1:
        st.markdown('<div class="section-header">ABC-XYZ Inventory Classification</div>', unsafe_allow_html=True)
        st.plotly_chart(_abc_heatmap(abc_df), use_container_width=True, key="inv_abc")

        a1, a2, a3 = st.columns(3)
        for col, cls, label, desc in [
            (a1,"A","Class A (High Revenue)","Top 70% revenue — maximum attention"),
            (a2,"B","Class B (Mid Revenue)", "70-90% revenue — regular review"),
            (a3,"C","Class C (Low Revenue)", "90-100% revenue — lean replenishment"),
        ]:
            count = abc_df[abc_df["abc_class"] == cls]["sku_count"].sum()
            col.metric(label, f"{count:,} SKUs", delta=desc)

    with tab2:
        st.markdown('<div class="section-header">Economic Order Quantity (EOQ) Calculator</div>', unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            demand      = st.number_input("Annual Demand (units)", 1, 500_000, 5000, 100, key="eoq_d")
            order_cost  = st.number_input("Order Cost per PO (£)", 1.0, 10_000.0, 150.0, 10.0, key="eoq_oc")
            holding_pct = st.slider("Holding Cost (% of unit cost/yr)", 1, 40, 20, key="eoq_hp")
            unit_cost   = st.number_input("Unit Cost (£)", 0.01, 5000.0, 25.0, 1.0, key="eoq_uc")
        with ec2:
            lead_time   = st.number_input("Lead Time (days)", 1, 180, 14, key="eoq_lt")
            svc         = st.selectbox("Service Level",
                                       ["90% (Z=1.28)","95% (Z=1.65)","98% (Z=2.05)","99% (Z=2.33)"],
                                       index=1, key="eoq_svc")
            z_map = {"90% (Z=1.28)":1.28,"95% (Z=1.65)":1.65,"98% (Z=2.05)":2.05,"99% (Z=2.33)":2.33}
            z = z_map[svc]

        result = _eoq(float(demand), float(order_cost), float(holding_pct),
                      float(unit_cost), int(lead_time), z)
        st.markdown("<br>", unsafe_allow_html=True)
        rc = st.columns(3)
        items = [
            ("📦 EOQ",               f"{result['eoq']:,.0f} units",  "Optimal order quantity"),
            ("🛡️ Safety Stock",       f"{result['ss']:,.0f} units",   f"At {svc.split('(')[0].strip()} service level"),
            ("🔁 Reorder Point",      f"{result['rop']:,.0f} units",  "Order when stock hits this level"),
            ("💰 Total Annual Cost",  f"£{result['tac']:,.0f}",       "Purchase + ordering + holding"),
            ("📋 Orders/Year",        f"{result['opy']:.1f}",         "Purchase orders per year"),
            ("📊 Avg Cycle Stock",    f"{result['acs']:,.0f} units",  "EOQ ÷ 2"),
        ]
        for i, (lbl, val, tip) in enumerate(items):
            with rc[i % 3]:
                st.markdown(
                    f'<div class="metric-card" style="text-align:center;padding:1rem;">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'<div class="metric-value" style="font-size:1.5rem;">{val}</div>'
                    f'<div class="tooltip-text">{tip}</div></div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown('<div class="section-header">🚨 Reorder Alerts</div>', unsafe_allow_html=True)
        ra1, ra2, ra3 = st.columns(3)
        ra1.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">Critical (≤7 days)</div><div style="font-size:2.2rem;font-weight:800;color:#dc2626;">{crit}</div></div>', unsafe_allow_html=True)
        ra2.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">High (≤14 days)</div><div style="font-size:2.2rem;font-weight:800;color:{PRIMARY};">{(reorder_df["urgency"]=="High").sum()}</div></div>', unsafe_allow_html=True)
        ra3.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">Medium (≤30 days)</div><div style="font-size:2.2rem;font-weight:800;color:{SECONDARY};">{(reorder_df["urgency"]=="Medium").sum()}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        urgency_bg = {"Critical":"#FEE2E2","High":"#FEF3C7","Medium":"#FEF9C3"}
        def _urg_style(row):
            bg = urgency_bg.get(row["urgency"], "#fff")
            return [f"background-color: {bg}" if c == "urgency" else "" for c in reorder_df.columns]

        styled = reorder_df.style.apply(_urg_style, axis=1)
        st.dataframe(styled, use_container_width=True, height=420,
                     column_config={
                         "current_stock":       st.column_config.NumberColumn("Stock",      format="%d"),
                         "reorder_point":       st.column_config.NumberColumn("ROP",        format="%d"),
                         "days_until_stockout": st.column_config.NumberColumn("Days",       format="%d"),
                     })
        st.download_button("⬇️ Download Reorder Alerts CSV",
                           reorder_df.to_csv(index=False).encode("utf-8-sig"),
                           f"NeuralRetail_ReorderAlerts_{date.today()}.csv",
                           "text/csv", key="dl_reorder")

    with tab4:
        st.markdown('<div class="section-header">Overstock Risk Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(_overstock_scatter(overstock_df), use_container_width=True, key="inv_os")
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("SKUs > 90 Days Supply", over90,  delta="Action required")
        ov2.metric("SKUs > £10k Holding",   (overstock_df["holding_cost"] > 10_000).sum())
        ov3.metric("High Dead Stock Risk",   dead,    delta="Clearance candidates")

    with tab5:
        st.markdown('<div class="section-header">Supplier Lead-Time Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(_supplier_box(supplier_df), use_container_width=True, key="inv_sup")

        sup_stats = (
            supplier_df.groupby("supplier")["lead_time_days"]
            .agg(["mean","std","min","max"]).reset_index()
        )
        sup_stats.columns = ["Supplier","Mean LT (days)","Std Dev","Min","Max"]
        sup_stats["CV (%)"] = (sup_stats["Std Dev"] / sup_stats["Mean LT (days)"] * 100).round(1)
        sup_stats["Flag"]   = sup_stats["CV (%)"].apply(lambda x: "⚠️ High Variance" if x > 30 else "✅ Stable")
        st.dataframe(sup_stats, use_container_width=True)


render()
