import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

from src.models.nlp_engine import NLPFinancialRiskEngine
from src.pipeline.feed_adapters import ReplayAdapter, RSSFeedAdapter
from src.pipeline.signal_store import RiskSignalStore
from src.rebalancer.rebalancer_engine import TacticalIndexRebalancer
from src.stress.stress_engine import WholesalePortfolioStressEngine

# Configure Streamlit page for wide Bloomberg/Institutional display
st.set_page_config(
    page_title="S&P x CRISIL | Enterprise Risk Intelligence Cockpit",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Institutional Wall Street / Bloomberg Terminal Style System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, .stCode, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Executive Top Navigation Header */
    .brand-header {
        background: linear-gradient(90deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 18px 24px;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .brand-title {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .system-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .system-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    /* Executive KPI Card Design */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #F9FAFB;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #6B7280;
        margin-top: 4px;
    }
    
    /* Institutional Article & Signal Card */
    .signal-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: border 0.2s ease;
    }
    .signal-card:hover {
        border-color: #38BDF8;
    }
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .signal-title {
        font-size: 16px;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 8px;
    }
    .signal-meta {
        font-size: 12px;
        color: #94A3B8;
        display: flex;
        gap: 16px;
    }
    
    /* Badges */
    .badge-event {
        background: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.2);
        color: #F97316;
        border: 1px solid rgba(249, 115, 22, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.2);
        color: #EAB308;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
    }
    
    /* Data Frame Clean Dark Theme */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1F2937 !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    # Loads FinBERT model pipeline
    return NLPFinancialRiskEngine(enable_transformer=True)

@st.cache_resource
def get_stress_engine():
    return WholesalePortfolioStressEngine()

@st.cache_resource
def get_rebalancer():
    return TacticalIndexRebalancer()

@st.cache_resource
def get_signal_store():
    return RiskSignalStore()

engine = get_engine()
stress_engine = get_stress_engine()
rebalancer = get_rebalancer()
store = get_signal_store()
@st.cache_resource
def get_user_profiles():
    with open("data/users_profiles.json", "r", encoding="utf-8") as f:
        return json.load(f)["users"]

user_profiles = get_user_profiles()
user_dict = {f"{u['display_name']} ({u['desk']})": u for u in user_profiles}

# Sidebar User Authentication & Desk Selector
st.sidebar.markdown("### 👤 User Authentication & Desk")
selected_user_label = st.sidebar.selectbox("Active Trader / Risk Officer:", list(user_dict.keys()))
active_user = user_dict[selected_user_label]

st.sidebar.markdown(f"""
<div style="background:#111827; border:1px solid #1F2937; border-radius:6px; padding:10px 12px; margin-bottom:12px;">
    <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase;">Authenticated Identity</div>
    <div style="font-size:13px; color:#F8FAFC; font-weight:700;">{active_user['display_name']}</div>
    <div style="font-size:11px; color:#38BDF8;">{active_user['role']}</div>
    <div style="font-size:11px; color:#64748B; margin-top:4px;">Watchlist: <code>{', '.join(active_user['watchlist'])}</code></div>
</div>
""", unsafe_allow_html=True)

# Top Corporate Branding Header
st.markdown(f"""
<div class="brand-header">
    <div>
        <div class="brand-title">🏛️ S&P Global & CRISIL | Risk Intelligence Platform</div>
        <div class="brand-subtitle">Enterprise AI/NLP Risk Engine & Event-Driven Capital Stress Testing System</div>
    </div>
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="text-align:right;">
            <div style="color:#F8FAFC; font-size:12px; font-weight:600;">{active_user['display_name']}</div>
            <div style="color:#94A3B8; font-size:11px;">{active_user['desk']}</div>
        </div>
        <div class="system-badge">
            <div class="system-pulse"></div>
            ACTIVE DESK
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Console
st.sidebar.markdown("### 🎛️ Risk Console")
app_mode = st.sidebar.radio(
    "Active Workspace",
    [
        "1. Real-Time Risk Engine (NLP Core)",
        "2. Module A: Tactical Index Rebalancer",
        "3. Module B: Wholesale Banking Stress Testing",
        "4. Institutional Model Governance (SR 11-7)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Data Feeds & Replay")
source_mode = st.sidebar.selectbox("Feed Ingestion Channel", ["Deterministic Crisis Replay", "Live RSS Feeds (CNBC/Yahoo)"])

if st.sidebar.button("⚡ Ingest & Process Batch", use_container_width=True, type="primary"):
    with st.spinner("Executing FinBERT transformer inference and taxonomy classification..."):
        if source_mode == "Deterministic Crisis Replay":
            adapter = ReplayAdapter()
        else:
            adapter = RSSFeedAdapter()
        articles = adapter.fetch_articles()
        signals = engine.process_batch(articles)
        st.sidebar.success(f"Ingested {len(signals)} verified signals.")

signals = store.get_all_signals(limit=100)

# -------------------------------------------------------------
# WORKSPACE 1: AI/NLP Risk Engine
# -------------------------------------------------------------
if app_mode == "1. Real-Time Risk Engine (NLP Core)":
    st.markdown("### ⚡ Live Financial NLP Risk Signals")
    st.caption("Central processing engine ingesting unstructured market feeds, computing FinBERT sentiment vectors, and outputting explainable 1–10 impact severity metrics.")

    # Executive KPI Stat Row
    critical_count = sum(1 for s in signals if s.impact_score >= 7.0)
    avg_sent = np.mean([s.sentiment_score for s in signals]) if signals else 0.0
    active_tickers = len(set(s.ticker for s in signals if s.ticker))

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Processed Stories</div>
            <div class="kpi-value">{len(signals)}</div>
            <div class="kpi-subtext">Deduplicated & Canonicalized</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High / Critical Signals</div>
            <div class="kpi-value" style="color: #EF4444;">{critical_count}</div>
            <div class="kpi-subtext">Severity Score ≥ 7.0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Aggregate Market Sentiment</div>
            <div class="kpi-value" style="color: {'#10B981' if avg_sent >= 0 else '#EF4444'};">{avg_sent:+.2f}</div>
            <div class="kpi-subtext">Scale: -1.00 (Fear) to +1.00 (Greed)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Active Securities Affected</div>
            <div class="kpi-value">{active_tickers}</div>
            <div class="kpi-subtext">Mapped to Security Master</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        min_impact_filter = st.slider("Filter by Minimum Impact Score", 1.0, 10.0, 1.0, 0.5)
    with f_col2:
        event_filter = st.selectbox("Filter Event Class", ["All Event Classes"] + sorted(list(set(s.event_type for s in signals))))
    with f_col3:
        user_watch_only = st.checkbox(f"🎯 Filter to {active_user['display_name']}'s Watchlist ({', '.join(active_user['watchlist'])})", value=False)

    filtered_signals = [
        s for s in signals 
        if s.impact_score >= min_impact_filter and 
           (event_filter == "All Event Classes" or s.event_type == event_filter) and
           (not user_watch_only or (s.ticker in active_user['watchlist'] or not s.ticker))
    ]

    st.markdown(f"**Showing {len(filtered_signals)} Signals (Filtered for {active_user['display_name']} • {active_user['desk']}):**")

    if not filtered_signals:
        st.info("No signals found matching criteria. Click '⚡ Ingest & Process Batch' in the sidebar.")
    else:
        for sig in filtered_signals:
            badge_color = {
                "CRITICAL": "badge-critical",
                "HIGH": "badge-high",
                "MEDIUM": "badge-medium",
                "LOW": "badge-low"
            }.get(sig.materiality, "badge-low")

            sent_color = "#10B981" if sig.sentiment_score > 0.15 else ("#EF4444" if sig.sentiment_score < -0.15 else "#94A3B8")

            with st.container():
                st.markdown(f"""
                <div class="signal-card">
                    <div class="signal-header">
                        <div>
                            <span class="badge-event">{sig.event_type}</span>
                            <span class="{badge_color}" style="margin-left: 8px;">SEVERITY {sig.impact_score:.1f}/10 • {sig.materiality}</span>
                        </div>
                        <div style="font-size: 12px; color: #64748B;">ID: <code>{sig.signal_id}</code></div>
                    </div>
                    <div class="signal-title">{sig.title}</div>
                    <div class="signal-meta">
                        <span>🏛️ Source: <strong>{sig.source_name}</strong></span>
                        <span>🎯 Entity: <strong>{f'{sig.company_name} ({sig.ticker})' if sig.ticker else 'Broad Market / Macro'}</strong></span>
                        <span>🕒 Published: <strong>{sig.published_at}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔍 Quantitative Model Breakdown & Traceable Text Evidence", expanded=False):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown("**Evidence Text Span Extracted:**")
                        st.info(f"\"{sig.evidence_text}\"")
                        st.markdown("**Impact Explanation Drivers:**")
                        for d in sig.impact_drivers:
                            st.write(f"• {d}")
                    with c2:
                        st.markdown("**FinBERT Sentiment Distribution:**")
                        st.markdown(f"<span style='font-size:22px; font-weight:700; color:{sent_color};'>{sig.sentiment_score:+.3f}</span> ({sig.sentiment_label})", unsafe_allow_html=True)
                        p_df = pd.DataFrame({
                            "State": ["Positive", "Neutral", "Negative"],
                            "Probability": [sig.sentiment_probs.get("positive", 0), sig.sentiment_probs.get("neutral", 0), sig.sentiment_probs.get("negative", 0)]
                        })
                        fig_p = px.bar(p_df, x="State", y="Probability", color="State", 
                                       color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"},
                                       height=180)
                        fig_p.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False, yaxis_range=[0, 1])
                        st.plotly_chart(fig_p, use_container_width=True)
                    with c3:
                        st.markdown("**Model Verification:**")
                        st.write(f"• Architecture: `{sig.model_version}`")
                        st.write(f"• Event Confidence: **{sig.event_confidence*100:.1f}%**")
                        st.write(f"• Impact Confidence: **{sig.impact_confidence*100:.1f}%**")
                        st.write(f"• Ingestion Trace: [External URL]({sig.url})")

# -------------------------------------------------------------
# WORKSPACE 2: Module A - Tactical Index Rebalancer
# -------------------------------------------------------------
elif app_mode == "2. Module A: Tactical Index Rebalancer":
    st.markdown("### 📊 Module A: Dynamic Tactical Index Rebalancing")
    st.caption("Quantitative asset allocation system subscribing to real-time NLP sentiment velocity to adjust S&P constituent weights while enforcing turnover constraints.")

    col_ctrl, col_stats = st.columns([1, 2])
    with col_ctrl:
        alpha = st.slider("Sentiment Tilt Sensitivity (α)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
        st.caption("Controls weight dispersion relative to equal-weight baseline.")
    
    rebalance_result = rebalancer.rebalance(signals, alpha=alpha)
    df_reb = pd.DataFrame([p.dict() for p in rebalance_result.positions])

    with col_stats:
        k1, k2, k3 = st.columns(3)
        k1.metric("Universe Constituents", len(rebalance_result.positions))
        k2.metric("Portfolio One-Way Turnover", f"{rebalance_result.total_turnover * 100:.2f}%")
        top_pick = rebalance_result.positions[0]
        k3.metric("Largest Overweight Asset", f"{top_pick.ticker} ({top_pick.target_weight*100:.1f}%)")

    st.markdown("---")
    
    # Visual Allocation Comparison Chart
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=df_reb["ticker"],
        y=df_reb["initial_weight"] * 100,
        name="Equal-Weight Benchmark (8.33%)",
        marker_color="#334155"
    ))
    fig_alloc.add_trace(go.Bar(
        x=df_reb["ticker"],
        y=df_reb["target_weight"] * 100,
        name="Tactical Sentiment Allocation (%)",
        marker_color="#0284C7"
    ))
    fig_alloc.update_layout(
        title="Asset Allocation Shift: Benchmark vs NLP Sentiment-Rebalanced Weights",
        xaxis_title="Security Symbol",
        yaxis_title="Weight Percentage (%)",
        barmode="group",
        height=400,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("#### 📋 Execution Ledger & Weight Deltas")
    display_df = df_reb[["ticker", "company_name", "sector", "current_sentiment", "initial_weight", "target_weight", "weight_delta"]].copy()
    display_df["initial_weight"] = (display_df["initial_weight"] * 100).round(2).astype(str) + "%"
    display_df["target_weight"] = (display_df["target_weight"] * 100).round(2).astype(str) + "%"
    display_df["weight_delta"] = (display_df["weight_delta"] * 100).map(lambda x: f"{x:+.2f}%")
    display_df["current_sentiment"] = display_df["current_sentiment"].map(lambda x: f"{x:+.3f}")
    
    st.dataframe(
        display_df.rename(columns={
            "ticker": "Ticker", "company_name": "Company Name", "sector": "Sector",
            "current_sentiment": "Sentiment Velocity", "initial_weight": "Benchmark Weight",
            "target_weight": "Target Weight", "weight_delta": "Tactical Delta"
        }),
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------------------
# WORKSPACE 3: Module B - Strategic Portfolio Stress Testing
# -------------------------------------------------------------
elif app_mode == "3. Module B: Wholesale Banking Stress Testing":
    st.markdown("### 🛡️ Module B: Event-Driven Wholesale Portfolio Stress Testing")
    st.caption("Basel-compliant simulation engine revaluing wholesale banking assets (Equities, Corporate Bonds, Commercial Loans, and Swaps) in response to high-impact textual risk events.")

    high_impact_signals = [s for s in signals if s.impact_score >= 6.5]

    st.markdown("#### 1. Real-Time Scenario Trigger Selection")
    if not high_impact_signals:
        st.warning("No high-impact signals found in the database. Ingest batch replay from the sidebar.")
        selected_signal = None
    else:
        sig_options = {f"[{s.event_type}] {s.title} (Severity: {s.impact_score:.1f}/10)": s for s in high_impact_signals}
        selected_title = st.selectbox("Select Real-World Event to Simulate:", list(sig_options.keys()))
        selected_signal = sig_options[selected_title]

    book_choice = st.radio(
        "Select Portfolio to Stress Test:",
        [f"👤 {active_user['display_name']}'s Book: {active_user['portfolio_name']}", "🏛️ Consolidated Wholesale Banking Book ($365M Bank-Wide)"],
        horizontal=True
    )
    is_user_book = "👤" in book_choice
    active_positions = active_user["positions"] if is_user_book else None

    if selected_signal:
        # Run stress test simulation on the selected user's book or bank-wide
        stress_res = stress_engine.run_stress_test(
            signal_title=selected_signal.title,
            event_type=selected_signal.event_type,
            impact_score=selected_signal.impact_score,
            confidence=selected_signal.impact_confidence,
            source_name=selected_signal.source_name,
            affected_ticker=selected_signal.ticker,
            custom_positions=active_positions
        )

        st.markdown("---")
        # Policy Gate Visual Box
        g_col1, g_col2 = st.columns([1, 2])
        with g_col1:
            if stress_res.policy_decision.accepted:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 6px; padding: 14px;">
                    <div style="color: #10B981; font-weight: 700; font-size: 13px; text-transform: uppercase;">✅ Policy Gate: ACCEPTED</div>
                    <div style="color: #E2E8F0; font-size: 12px; margin-top: 6px;">{stress_res.policy_decision.trigger_reason}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 6px; padding: 14px;">
                    <div style="color: #EF4444; font-weight: 700; font-size: 13px; text-transform: uppercase;">❌ Policy Gate: REJECTED</div>
                    <div style="color: #E2E8F0; font-size: 12px; margin-top: 6px;">{stress_res.policy_decision.trigger_reason}</div>
                </div>
                """, unsafe_allow_html=True)
        with g_col2:
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 6px; padding: 14px;">
                <div style="color: #38BDF8; font-weight: 600; font-size: 13px;">Activated Stress Scenario: {stress_res.scenario_name}</div>
                <div style="color: #94A3B8; font-size: 12px; margin-top: 6px;">
                    {' • '.join(stress_res.assumptions)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Top Portfolio Balance Sheet Metrics
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Baseline Book Value</div>
                <div class="kpi-value">${stress_res.total_baseline_portfolio_value/1e6:.1f}M</div>
                <div class="kpi-subtext">Pre-Stress NAV ($365.0M Book)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Stressed Book Value</div>
                <div class="kpi-value">${stress_res.total_stressed_portfolio_value/1e6:.1f}M</div>
                <div class="kpi-subtext">Post-Shock Net Asset Value</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Estimated P&L Loss</div>
                <div class="kpi-value" style="color: #EF4444;">-${stress_res.total_pnl_loss/1e6:.2f}M</div>
                <div class="kpi-subtext">Portfolio Impact: {stress_res.total_percentage_loss:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Dominant Risk Driver</div>
                <div class="kpi-value" style="font-size: 18px; color: #F97316;">{max(stress_res.asset_class_losses.items(), key=lambda x: x[1])[0]}</div>
                <div class="kpi-subtext">Largest Asset-Class Loss Contributor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts Row
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_waterfall = go.Figure(go.Waterfall(
                name="Capital Flow",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=["Baseline NAV", "Equity Loss", "Bond Loss", "Loan Provision", "Swap Loss", "Stressed NAV"],
                textposition="outside",
                text=[
                    f"${stress_res.total_baseline_portfolio_value/1e6:.1f}M",
                    f"-${stress_res.asset_class_losses['Equity']/1e6:.2f}M",
                    f"-${stress_res.asset_class_losses['Bond']/1e6:.2f}M",
                    f"-${stress_res.asset_class_losses['Loan']/1e6:.2f}M",
                    f"-${stress_res.asset_class_losses['Swap']/1e6:.2f}M",
                    f"${stress_res.total_stressed_portfolio_value/1e6:.1f}M"
                ],
                y=[
                    stress_res.total_baseline_portfolio_value,
                    -stress_res.asset_class_losses['Equity'],
                    -stress_res.asset_class_losses['Bond'],
                    -stress_res.asset_class_losses['Loan'],
                    -stress_res.asset_class_losses['Swap'],
                    stress_res.total_stressed_portfolio_value
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#EF4444"}},
                totals={"marker": {"color": "#0284C7"}},
                increasing={"marker": {"color": "#10B981"}}
            ))
            fig_waterfall.update_layout(
                title="Wholesale Portfolio Stress Loss Waterfall ($)",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with col_c2:
            pie_data = [{"Asset Class": k, "Loss": v} for k, v in stress_res.asset_class_losses.items() if v > 0]
            fig_pie = px.pie(
                pd.DataFrame(pie_data),
                names="Asset Class",
                values="Loss",
                title="Stress Loss Attribution by Asset Class",
                color_discrete_sequence=["#EF4444", "#F97316", "#EAB308", "#38BDF8"],
                hole=0.45,
                template="plotly_dark"
            )
            fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📑 Granular Position Revaluation Matrix")
        df_pos = pd.DataFrame([p.dict() for p in stress_res.position_impacts])
        disp_pos = df_pos[["position_id", "asset_type", "ticker", "issuer", "baseline_value", "stressed_value", "pnl_loss", "percentage_change", "applied_shock"]].copy()
        disp_pos["baseline_value"] = disp_pos["baseline_value"].map("${:,.0f}".format)
        disp_pos["stressed_value"] = disp_pos["stressed_value"].map("${:,.0f}".format)
        disp_pos["pnl_loss"] = disp_pos["pnl_loss"].map(lambda x: f"-${x:,.0f}" if x > 0 else "$0")
        disp_pos["percentage_change"] = disp_pos["percentage_change"].map("{:.2f}%".format)

        st.dataframe(
            disp_pos.rename(columns={
                "position_id": "Position ID", "asset_type": "Asset Type", "ticker": "Ticker",
                "issuer": "Issuer / Desk", "baseline_value": "Baseline Value", "stressed_value": "Stressed Value",
                "pnl_loss": "Stress P&L Loss", "percentage_change": "% Change", "applied_shock": "Applied Valuation Sensitivity"
            }),
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------------------
# WORKSPACE 4: Model Governance & SR 11-7 Controls
# -------------------------------------------------------------
elif app_mode == "4. Institutional Model Governance (SR 11-7)":
    st.markdown("### 📜 Institutional Model Governance & Regulatory Audit (SR 11-7 / BCBS 239)")
    st.caption("Demonstrating model validation, sound stress testing principles, and explicit handling of false positive / non-material noise.")

    gov_c1, gov_c2 = st.columns(2)
    with gov_c1:
        st.markdown("#### 🛡️ False Positive Audit Test")
        st.write("""
        Federal Reserve Supervisory Letter **SR 11-7** dictates that risk engines must possess robust operational controls preventing low-materiality noise from triggering spurious capital decisions.
        """)

        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:16px;">
            <div style="font-weight:600; color:#F8FAFC; margin-bottom:8px;">Test Case: Cosmetic Product Rumor ("Apple pastel watch strap color options")</div>
            <div style="font-size:12px; color:#94A3B8; margin-bottom:12px;">Simulated input from unverified blog source with non-material subject matter.</div>
            <div style="display:flex; gap:16px; font-size:13px;">
                <span>Impact Score: <strong>2.64 / 10</strong></span>
                <span>Confidence: <strong>55%</strong></span>
                <span>Decision: <strong style="color:#EF4444;">REJECTED</strong></span>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#64748B;">Audit Reason: <em>Impact score below systemic threshold of 7.0 (Non-material noise).</em></div>
        </div>
        """, unsafe_allow_html=True)

    with gov_c2:
        st.markdown("#### 📐 Institutional Valuation Methodology")
        st.markdown("""
        | Asset Class | Mathematical Sensitivity Model | Regulatory Standard |
        | :--- | :--- | :--- |
        | **Equities** | $\\Delta P = MV \\times \\beta \\times \\Delta R_{\\text{market}}$ | Single-Factor CAPM Beta |
        | **Bonds** | $\\Delta P \\approx -\\text{Duration} \\times (\\Delta y + \\Delta s)$ | Modified Duration & Credit Spread |
        | **Wholesale Loans** | $\\Delta EL = EAD \\times (PD_{\\text{stress}}LGD_{\\text{stress}} - PD_0 LGD_0)$ | Basel Advanced IRB Framework |
        | **Interest Swaps** | $\\Delta P = -DV01 \\times \\Delta r$ | Delta-Normal DV01 Rate Sensitivity |
        """)

    st.markdown("---")
    st.markdown("#### 🔍 Active Model Lineage & System Audit Trail")
    st.markdown("""
    - **FinBERT Weights:** `ProsusAI/finbert` (Hugging Face Transformers / PyTorch sequence classification).
    - **Entity Master:** 15 Corporate Issuers with curated aliases, security tickers, and credit ratings.
    - **Event Taxonomies:** 8-Class hierarchical taxonomy covering Credit Events, Macro, Geopolitical, M&A, Regulatory, Supply Chain, Cyber, and Product Launches.
    - **Persistence:** SQLite immutable append-only risk ledger ensuring complete BCBS 239 auditability from raw article to stressed P&L.
    """)
