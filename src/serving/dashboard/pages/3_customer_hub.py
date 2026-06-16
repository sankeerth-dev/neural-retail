"""Page 3 — Customer Hub Dashboard.

NeuralRetail Intelligence Platform · AMX-DS-2026-04
Customer 360 view, churn risk scoring, RFM segment radar,
CLV tier distribution, and retention action recommendations.
All data synthesised in-process.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PRIMARY   = "#E84E1B"
SECONDARY = "#F7941D"
ACCENT    = "#FBBA13"
_CSS = Path(__file__).parents[1] / "assets" / "style.css"

_PERSONAS = ["Champion", "Loyal", "Potential Loyalist", "At-Risk VIP", "New", "Lost"]
_PERSONA_COLORS = {
    "Champion":          ("#FEF3C7","#92400E"),
    "Loyal":             ("#DCFCE7","#166534"),
    "Potential Loyalist":("#DBEAFE","#1E40AF"),
    "At-Risk VIP":       ("#FEE2E2","#991B1B"),
    "New":               ("#E0F2FE","#075985"),
    "Lost":              ("#F3F4F6","#374151"),
}
_CLV_TIERS = {"Platinum": PRIMARY, "Gold": SECONDARY, "Silver": ACCENT, "Bronze": "#94A3B8"}


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
        st.warning("🔒 Analyst or Admin role required for the Customer Hub.")
        return False
    return True


# ── Data generators ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _customers(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    personas = rng.choice(_PERSONAS, n, p=[.12,.18,.22,.15,.20,.13])
    clv_map  = {"Champion":"Platinum","Loyal":"Gold","Potential Loyalist":"Silver",
                "At-Risk VIP":"Gold","New":"Bronze","Lost":"Bronze"}
    recency  = rng.uniform(1, 180, n)
    freq     = rng.integers(1, 50, n).astype(float)
    monetary = rng.uniform(20, 6000, n)
    # Churn proba driven by persona
    base_churn = {"Champion":.05,"Loyal":.10,"Potential Loyalist":.25,
                  "At-Risk VIP":.72,"New":.30,"Lost":.88}
    churn_p = np.array([base_churn[p] + rng.uniform(-.05,.05) for p in personas]).clip(0,1)
    risk_tier = pd.cut(churn_p, [0,.4,.6,.8,1], labels=["Low","Medium","High","Critical"])
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "persona":     personas,
        "clv_tier":    [clv_map[p] for p in personas],
        "recency_days": recency.round(0).astype(int),
        "frequency":   freq,
        "monetary":    monetary.round(2),
        "churn_proba": churn_p.round(4),
        "risk_tier":   risk_tier,
        "last_purchase": [str(date.today() - timedelta(days=int(r))) for r in recency],
    })


@st.cache_data(ttl=300)
def _segment_profiles() -> pd.DataFrame:
    return pd.DataFrame({
        "persona":          _PERSONAS,
        "recency_norm":     [90, 75, 55, 35, 45, 15],
        "frequency_norm":   [88, 80, 60, 40, 25, 10],
        "monetary_norm":    [92, 78, 55, 65, 30, 12],
        "engagement_norm":  [85, 72, 58, 30, 50, 8],
        "satisfaction_norm":[88, 80, 62, 40, 60, 15],
    })


@st.cache_data(ttl=300)
def _churn_heatmap_data() -> pd.DataFrame:
    rows = []
    for persona in _PERSONAS:
        for decile in range(1, 11):
            rows.append({"persona": persona, "risk_decile": decile,
                         "customer_count": max(0, int(np.random.default_rng(hash(persona)*decile % 2**31).integers(5, 80)))})
    return pd.DataFrame(rows)


def _txn_timeline(customer_id: str) -> pd.DataFrame:
    seed = hash(customer_id) % 2**31
    rng  = np.random.default_rng(seed)
    n    = int(rng.integers(8, 30))
    cats = rng.choice(["Electronics","Apparel","Food","Health","Home"], n)
    days_back = np.sort(rng.integers(1, 180, n))[::-1]
    return pd.DataFrame({
        "date":     [date.today() - timedelta(days=int(d)) for d in days_back],
        "category": cats,
        "amount":   rng.uniform(15, 400, n).round(2),
        "basket":   rng.integers(1, 8, n),
    })


# ── Charts ────────────────────────────────────────────────────────────────
def _segment_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["persona"].value_counts().reset_index()
    counts.columns = ["persona", "count"]
    fig = go.Figure(go.Pie(
        labels=counts["persona"], values=counts["count"],
        hole=0.52, textinfo="label+percent",
        marker=dict(colors=[PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7", "#94A3B8"]),
    ))
    fig.update_layout(title="Segment Distribution", showlegend=True,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, sans-serif", size=12, color="#111827"),
                      title_font=dict(color="#111827"),
                      legend=dict(font=dict(color="#111827")),
                      margin=dict(l=20, r=20, t=50, b=20), height=370)
    return fig


def _clv_bar(df: pd.DataFrame) -> go.Figure:
    clv_df = df.groupby("clv_tier")["monetary"].sum().reset_index()
    clv_df.columns = ["tier", "revenue"]
    order = ["Platinum","Gold","Silver","Bronze"]
    clv_df["order"] = clv_df["tier"].map({t:i for i,t in enumerate(order)})
    clv_df = clv_df.sort_values("order")
    fig = go.Figure(go.Bar(
        x=clv_df["tier"], y=clv_df["revenue"],
        marker=dict(color=[_CLV_TIERS[t] for t in clv_df["tier"]]),
        text=[f"£{v/1e6:.2f}M" for v in clv_df["revenue"]],
        textposition="outside",
    ))
    fig.update_layout(title="CLV Tier Revenue Contribution",
                      yaxis=dict(tickprefix="£", tickformat=",.0f", gridcolor="#f5f5f5",
                                 tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
                      xaxis=dict(tickfont=dict(color="#111827")),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, sans-serif", size=12, color="#111827"),
                      title_font=dict(color="#111827"),
                      margin=dict(l=50, r=20, t=50, b=40))
    return fig


def _radar_fig(profiles_df: pd.DataFrame, selected: list[str]) -> go.Figure:
    axes = ["recency_norm","frequency_norm","monetary_norm","engagement_norm","satisfaction_norm"]
    labels = ["Recency","Frequency","Monetary","Engagement","Satisfaction"]
    colors = [PRIMARY, SECONDARY, ACCENT, "#22C55E", "#885CF7", "#94A3B8"]
    fig = go.Figure()
    for i, persona in enumerate(selected):
        row = profiles_df[profiles_df["persona"] == persona]
        if row.empty:
            continue
        vals = [float(row[ax].iloc[0]) for ax in axes] + [float(row[axes[0]].iloc[0])]
        h = colors[i % len(colors)].lstrip('#')
        rgba_color = f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},0.13)"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels + [labels[0]], fill="toself", name=persona,
            line=dict(color=colors[i % len(colors)], width=2.5),
            fillcolor=rgba_color,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#eee",
                                   tickfont=dict(color="#111827")),
                   angularaxis=dict(gridcolor="#eee", tickfont=dict(color="#111827"))),
        title="Segment RFM Radar", showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#111827"),
        title_font=dict(color="#111827"),
        margin=dict(l=40, r=40, t=50, b=60), height=420,
    )
    return fig


def _churn_heatmap_fig(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot(index="risk_decile", columns="persona", values="customer_count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=[f"D{d}" for d in pivot.index],
        colorscale=[[0,"#DCFCE7"],[0.5,ACCENT],[1.0,PRIMARY]],
        text=pivot.values.astype(int), texttemplate="%{text}",
        hovertemplate="Persona: %{x}<br>Decile: %{y}<br>Customers: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Churn Risk Heatmap (Segment × Risk Decile)",
                      xaxis_title="Persona", yaxis_title="Risk Decile",
                      xaxis=dict(tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
                      yaxis=dict(tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
                      font=dict(family="Inter, sans-serif", size=11, color="#111827"),
                      title_font=dict(color="#111827"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=50, r=20, t=50, b=60))
    return fig


def _timeline_fig(df: pd.DataFrame, cid: str) -> go.Figure:
    cat_colors = {"Electronics":PRIMARY,"Apparel":SECONDARY,"Food":ACCENT,
                  "Health":"#22C55E","Home":"#885CF7"}
    fig = go.Figure()
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["amount"], mode="markers",
            name=cat, marker=dict(color=cat_colors.get(cat,"#888"),
                                  size=sub["basket"]*4+6, opacity=0.85,
                                  line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{cat}</b><br>Date: %{{x}}<br>£%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Purchase Timeline — {cid}",
        xaxis=dict(gridcolor="#f5f5f5",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        yaxis=dict(gridcolor="#f5f5f5", title="Amount (£)",
                   tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#111827"),
        title_font=dict(color="#111827"),
        legend=dict(orientation="h", y=-0.2, font=dict(color="#111827")),
        margin=dict(l=50, r=20, t=50, b=70),
    )
    return fig


def _risk_badge(p: float) -> str:
    if p > .80: return f'<span style="background:#dc2626;color:white;padding:3px 10px;border-radius:5px;font-size:.8rem;font-weight:700;">Critical ({p:.1%})</span>'
    if p > .60: return f'<span style="background:{PRIMARY};color:white;padding:3px 10px;border-radius:5px;font-size:.8rem;font-weight:700;">High ({p:.1%})</span>'
    if p > .40: return f'<span style="background:{SECONDARY};color:white;padding:3px 10px;border-radius:5px;font-size:.8rem;font-weight:700;">Medium ({p:.1%})</span>'
    return f'<span style="background:#16a34a;color:white;padding:3px 10px;border-radius:5px;font-size:.8rem;font-weight:700;">Low ({p:.1%})</span>'


def _persona_badge(persona: str) -> str:
    bg, tc = _PERSONA_COLORS.get(persona, ("#f0f0f0","#555"))
    return f'<span style="background:{bg};color:{tc};padding:3px 12px;border-radius:20px;font-size:.8rem;font-weight:700;">{persona}</span>'


def _retention_actions(persona: str, churn_p: float) -> list[dict]:
    actions_map = {
        "At-Risk VIP": [
            {"action":"Send personalised win-back email with 20% exclusive discount","urgency":"High","channel":"Email","lift":"15-22%"},
            {"action":"Assign VIP account manager for proactive outreach call","urgency":"High","channel":"Phone","lift":"12-18%"},
        ],
        "Loyal": [
            {"action":"Award double loyalty points on next purchase to reward frequency","urgency":"Medium","channel":"Push","lift":"8-12%"},
        ],
        "Lost": [
            {"action":"Launch reactivation SMS: 30-day limited 25% discount","urgency":"Critical","channel":"SMS","lift":"6-10%"},
        ],
        "Potential Loyalist": [
            {"action":"Invite to loyalty programme with sign-up bonus offer","urgency":"Medium","channel":"Email","lift":"10-16%"},
        ],
    }
    defaults = [{"action":"Send re-engagement push notification with personalised offer","urgency":"Medium","channel":"Push","lift":"5-9%"}]
    return actions_map.get(persona, defaults if churn_p > 0.4 else [])


# ── Page ──────────────────────────────────────────────────────────────────
def render() -> None:
    _css()
    if not _check_auth():
        return

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">👥 Customer Hub</div>'
        '<div class="page-subtitle">Customer 360 view, segment analysis, churn risk scoring, and retention actions</div>',
        unsafe_allow_html=True,
    )

    cust_df  = _customers()
    prof_df  = _segment_profiles()
    heat_df  = _churn_heatmap_data()

    # Sidebar
    with st.sidebar:
        st.markdown("### Filters")
        persona_filter = st.multiselect("Segment Filter", _PERSONAS, default=_PERSONAS, key="ch_seg")
        risk_filter    = st.multiselect("Risk Tier", ["Critical","High","Medium","Low"],
                                        default=["Critical","High","Medium","Low"], key="ch_risk")

    filtered = cust_df[
        cust_df["persona"].isin(persona_filter) & cust_df["risk_tier"].isin(risk_filter)
    ]

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",      f"{len(cust_df):,}")
    k2.metric("High/Critical Risk",   f"{(cust_df['churn_proba'] > .6).sum():,}",
              delta=f"{(cust_df['churn_proba'] > .6).mean()*100:.1f}%")
    k3.metric("Avg Churn Probability", f"{cust_df['churn_proba'].mean():.3f}")
    k4.metric("Avg Customer LTV",     f"£{cust_df['monetary'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Segment Overview", "👤 Customer 360", "🔥 Churn Heatmap", "🛡️ Retention Actions"]
    )

    with tab1:
        st.markdown('<div class="section-header">Segment Distribution & CLV</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.plotly_chart(_segment_donut(filtered), use_container_width=True, key="ch_donut")
        with sc2:
            st.plotly_chart(_clv_bar(filtered), use_container_width=True, key="ch_clv")

        st.markdown('<div class="section-header">RFM Segment Radar</div>', unsafe_allow_html=True)
        selected_personas = st.multiselect(
            "Overlay Segments", _PERSONAS, default=["Champion","At-Risk VIP","Lost"], key="ch_radar_sel"
        )
        if selected_personas:
            st.plotly_chart(_radar_fig(prof_df, selected_personas), use_container_width=True, key="ch_radar")

    with tab2:
        st.markdown('<div class="section-header">Individual Customer Lookup</div>', unsafe_allow_html=True)
        cid_list = filtered.sort_values("churn_proba", ascending=False)["customer_id"].tolist()
        cid = st.selectbox("Select Customer", cid_list[:100], key="ch_cid")

        cust_row = filtered[filtered["customer_id"] == cid].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-card"><div class="metric-label">Segment</div>'
            f'{_persona_badge(cust_row["persona"])}</div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-card"><div class="metric-label">Churn Risk</div>'
            f'{_risk_badge(cust_row["churn_proba"])}</div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="metric-card"><div class="metric-label">CLV Tier</div>'
            f'<div class="metric-value" style="font-size:1.3rem;">{cust_row["clv_tier"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        det1, det2, det3, det4 = st.columns(4)
        det1.metric("Recency",    f"{int(cust_row['recency_days'])} days")
        det2.metric("Frequency",  f"{int(cust_row['frequency'])} orders")
        det3.metric("Monetary",   f"£{cust_row['monetary']:,.2f}")
        det4.metric("Last Purchase", cust_row["last_purchase"])

        txn_df = _txn_timeline(cid)
        st.plotly_chart(_timeline_fig(txn_df, cid), use_container_width=True, key="ch_txn")

    with tab3:
        st.markdown('<div class="section-header">Churn Risk Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(_churn_heatmap_fig(heat_df), use_container_width=True, key="ch_heat")

        high_risk = cust_df[cust_df["churn_proba"] > 0.6].sort_values("churn_proba", ascending=False)
        st.markdown(f"**Top 20 Highest-Risk Customers ({len(high_risk)} total high-risk)**")
        st.dataframe(
            high_risk.head(20)[["customer_id","persona","clv_tier","churn_proba","recency_days","monetary"]],
            use_container_width=True,
            column_config={
                "churn_proba":  st.column_config.ProgressColumn("Churn Proba", min_value=0, max_value=1, format="%.3f"),
                "monetary":     st.column_config.NumberColumn("LTV (£)", format="£%.2f"),
                "recency_days": st.column_config.NumberColumn("Recency (days)"),
            },
        )
        st.download_button(
            "⬇️ Download High-Risk List CSV",
            high_risk.to_csv(index=False).encode("utf-8-sig"),
            f"NeuralRetail_HighRisk_{date.today()}.csv",
            "text/csv", key="dl_hr",
        )

    with tab4:
        st.markdown('<div class="section-header">Retention Action Recommendations</div>', unsafe_allow_html=True)
        act_persona = st.selectbox("Persona", _PERSONAS, index=3, key="ch_act_persona")
        act_churn   = st.slider("Representative Churn Probability", 0.0, 1.0, 0.72, 0.01, key="ch_act_churn")

        actions = _retention_actions(act_persona, act_churn)
        if actions:
            for a in actions:
                urgency_color = {"Critical":"#dc2626","High":PRIMARY,"Medium":SECONDARY}.get(a["urgency"], ACCENT)
                st.markdown(
                    f"""
                    <div class="action-card">
                        <div style="display:flex;justify-content:space-between;margin-bottom:.4rem;">
                            <span style="font-weight:700;color:{urgency_color};">
                                {a['urgency']} Urgency — {a['channel']}
                            </span>
                            <span style="font-size:.8rem;color:#16a34a;font-weight:600;">
                                Expected Lift: {a['lift']}
                            </span>
                        </div>
                        <div style="font-size:.88rem;color:#444;">{a['action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No specific retention action recommended for this segment at current risk level.")

        # Segment-level bulk export
        seg_customers = cust_df[cust_df["persona"] == act_persona]
        st.markdown(f"**{len(seg_customers)} customers in '{act_persona}' segment**")
        st.download_button(
            f"⬇️ Export {act_persona} Segment CSV",
            seg_customers.to_csv(index=False).encode("utf-8-sig"),
            f"NeuralRetail_{act_persona.replace(' ','_')}_{date.today()}.csv",
            "text/csv", key="dl_seg",
        )


render()
