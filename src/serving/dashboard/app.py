"""NeuralRetail Intelligence Platform — Streamlit Entry Point.

Self-contained multi-page dashboard. No external auth libraries required.
Login is handled via st.session_state with a simple credential map.

Run:
    streamlit run src/serving/dashboard/app.py

Demo credentials:
    admin   / admin123   (Admin  — all pages)
    analyst / analyst123 (Analyst — pages 1-4)
    viewer  / viewer123  (Viewer  — pages 1-2)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ── Page config (MUST be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="NeuralRetail Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────
_ASSETS = Path(__file__).parent / "assets"
_CSS    = _ASSETS / "style.css"

# ── Demo credentials {username: (password, display_name, role)} ──────────
_CREDENTIALS: dict[str, tuple[str, str, str]] = {
    "admin":   ("admin123",   "Admin User",     "Admin"),
    "analyst": ("analyst123", "Data Analyst",   "Analyst"),
    "viewer":  ("viewer123",  "Business Viewer","Viewer"),
}

# ── Page access control ──────────────────────────────────────────────────
_PAGE_ACCESS: dict[str, list[str]] = {
    "1_executive_kpi":      ["Admin", "Analyst", "Viewer"],
    "2_demand_intelligence":["Admin", "Analyst", "Viewer"],
    "3_customer_hub":       ["Admin", "Analyst"],
    "4_inventory_health":   ["Admin", "Analyst"],
    "5_mlops_monitor":      ["Admin"],
}
_PAGE_ICONS  = {"1_executive_kpi":"📊","2_demand_intelligence":"📈",
                "3_customer_hub":"👥","4_inventory_health":"📦","5_mlops_monitor":"🔬"}
_PAGE_LABELS = {"1_executive_kpi":"Executive KPI","2_demand_intelligence":"Demand Intelligence",
                "3_customer_hub":"Customer Hub","4_inventory_health":"Inventory Health",
                "5_mlops_monitor":"MLOps Monitor"}
_PAGE_DESCS  = {
    "1_executive_kpi":      "Revenue trends, churn gauges & board-ready KPI summaries.",
    "2_demand_intelligence":"SKU-level demand forecasts with what-if scenario simulator.",
    "3_customer_hub":       "Customer 360 view, segment radar & retention actions.",
    "4_inventory_health":   "ABC-XYZ matrix, EOQ calculator & reorder alerts.",
    "5_mlops_monitor":      "Drift PSI bars, model registry & retrain pipeline controls.",
}
_ROLE_COLORS = {"Admin":"#E84E1B","Analyst":"#F7941D","Viewer":"#FBBA13"}


# ── CSS helper ───────────────────────────────────────────────────────────
def _inject_css() -> None:
    if _CSS.exists():
        st.markdown(f"<style>{_CSS.read_text('utf-8')}</style>", unsafe_allow_html=True)


# ── Session-state init ───────────────────────────────────────────────────
def _init_state() -> None:
    for k, v in [
        ("logged_in", False), ("username", None), ("display_name", None), ("role", None),
        ("selected_sku", "SKU-1001"), ("selected_segment", "All"),
        ("date_range_days", 90), ("selected_store", "All Stores"),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


# ── Login screen ─────────────────────────────────────────────────────────
def _render_login() -> None:
    _inject_css()
    # Centered hero
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown(
            """
            <div class="header-bar"></div>
            <div style="text-align:center; padding:2.5rem 0 1.8rem;">
                <span style="font-size:3rem;font-weight:900;color:#E84E1B;">Neural</span><span style="font-size:3rem;font-weight:900;color:#F7941D;">Retail</span>
                <p style="color:#6b7280;font-size:0.85rem;letter-spacing:.12em;margin:6px 0 0;">
                    INTELLIGENCE PLATFORM &middot; AMX-DS-2026-04
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Login card
        st.markdown(
            """
            <div class="login-card">
                <div class="login-card-title">&#128272; Sign In</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uname = st.text_input("Username", placeholder="admin / analyst / viewer", key="login_user")
        pwd   = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd")

        if st.button("Sign In →", type="primary", use_container_width=True, key="login_btn"):
            creds = _CREDENTIALS.get(uname)
            if creds and creds[0] == pwd:
                st.session_state.logged_in    = True
                st.session_state.username     = uname
                st.session_state.display_name = creds[1]
                st.session_state.role         = creds[2]
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Try admin / admin123")




# ── Sidebar (authenticated) ──────────────────────────────────────────────
def _render_sidebar() -> None:
    role  = st.session_state.role
    uname = st.session_state.display_name
    color = _ROLE_COLORS.get(role, "#888")

    with st.sidebar:
        st.markdown(
            """
            <div class="header-bar"></div>
            <div style="text-align:center;padding:.5rem 0 1.2rem;">
                <span style="font-size:1.6rem;font-weight:900;color:#E84E1B;">Neural</span>
                <span style="font-size:1.6rem;font-weight:900;color:#F7941D;">Retail</span>
                <p style="font-size:.65rem;color:#888;letter-spacing:.12em;margin:2px 0 0;">
                    INTELLIGENCE PLATFORM
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Navigation**")
        for page_key, allowed in _PAGE_ACCESS.items():
            if role in allowed:
                icon  = _PAGE_ICONS[page_key]
                label = _PAGE_LABELS[page_key]
                st.page_link(f"pages/{page_key}.py", label=f"{icon}  {label}")

        st.divider()

        # User badge
        st.markdown(
            f"""
            <div style="padding:.75rem;background:#1a1a1a;border-radius:9px;border:1px solid #333;">
                <p style="margin:0;font-size:.72rem;color:#888;">Signed in as</p>
                <p style="margin:3px 0 6px;font-weight:700;font-size:.92rem;color:#e5e5e5;">{uname}</p>
                <span style="background:{color}22;color:{color};padding:2px 9px;
                             border-radius:4px;font-size:.7rem;font-weight:700;">
                    {role.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Out", key="logout_btn", use_container_width=True):
            for k in ["logged_in","username","display_name","role"]:
                st.session_state[k] = None if k != "logged_in" else False
            st.rerun()


# ── Landing page ─────────────────────────────────────────────────────────
def _render_landing() -> None:
    _inject_css()
    _render_sidebar()

    role  = st.session_state.role
    uname = st.session_state.display_name

    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h1 style="font-size:2rem;font-weight:900;color:#1a1a1a;margin-bottom:.15rem;">
            Welcome back, {uname} 👋
        </h1>
        <p style="color:#888;font-size:.9rem;margin:0 0 1.8rem;">
            NeuralRetail Intelligence Platform — AMX-DS-2026-04
        </p>
        """,
        unsafe_allow_html=True,
    )

    allowed = [p for p, roles in _PAGE_ACCESS.items() if role in roles]

    # ── KPI summary row ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, delta, pos in [
        (k1, "£2.84M",  "Monthly Revenue",   "+12.3% MoM",  True),
        (k2, "8.7%",    "Demand MAPE",       "↓ 1.3pp vs target", True),
        (k3, "0.921",   "Churn AUC-ROC",     "+0.011 vs baseline", True),
        (k4, "4.2%",    "Stockout Rate",     "▼ 0.8pp WoW", True),
    ]:
        delta_cls = "metric-delta-positive" if pos else "metric-delta-negative"
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="{delta_cls}">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Quick Access</div>', unsafe_allow_html=True)

    # ── Page cards ──────────────────────────────────────────────────────
    n = len(allowed)
    cols = st.columns(min(n, 3))
    for i, page_key in enumerate(allowed[:3]):
        with cols[i]:
            icon  = _PAGE_ICONS[page_key]
            label = _PAGE_LABELS[page_key]
            desc  = _PAGE_DESCS[page_key]
            st.markdown(
                f"""
                <div class="metric-card" style="min-height:120px; margin-bottom: 0.5rem;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-weight:800;font-size:.95rem;margin:.4rem 0 .2rem;color:#111827;">{label}</div>
                    <div style="font-size:.82rem;color:#6b7280;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"btn_{page_key}", use_container_width=True):
                st.switch_page(f"pages/{page_key}.py")

    if n > 3:
        cols2 = st.columns(n - 3)
        for i, page_key in enumerate(allowed[3:]):
            with cols2[i]:
                icon  = _PAGE_ICONS[page_key]
                label = _PAGE_LABELS[page_key]
                desc  = _PAGE_DESCS[page_key]
                st.markdown(
                    f"""
                    <div class="metric-card" style="min-height:120px; margin-bottom: 0.5rem;">
                        <div style="font-size:2rem;">{icon}</div>
                        <div style="font-weight:800;font-size:.95rem;margin:.4rem 0 .2rem;color:#111827;">{label}</div>
                        <div style="font-size:.82rem;color:#6b7280;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {label}", key=f"btn_{page_key}", use_container_width=True):
                    st.switch_page(f"pages/{page_key}.py")

    # ── Platform status strip ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Platform Status</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, label, badge, cls in [
        (s1, "demand_ensemble",          "Production v3",    "status-badge-pass"),
        (s2, "churn_stacking_ensemble",  "Production v5",    "status-badge-pass"),
        (s3, "kmeans_segmentation",      "Production v2",    "status-badge-pass"),
        (s4, "Evidently Drift Monitor",  "No Drift Detected","status-badge-pass"),
        (s5, "FastAPI Scoring API",      "P95 < 0.9s",       "status-badge-pass"),
    ]:
        col.markdown(
            f"""
            <div class="metric-card" style="text-align:center;padding:1rem;">
                <div style="font-size:0.72rem;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;">{label}</div>
                <span class="{cls}">{badge}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    _init_state()

    if not st.session_state.logged_in:
        _render_login()
    else:
        _render_landing()


if __name__ == "__main__":
    main()
