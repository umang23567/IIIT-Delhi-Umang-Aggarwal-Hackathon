"""Streamlit dashboard.  Run: streamlit run src/dashboard.py   (after: python main.py all)"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.modules.stress import run_stress  # noqa: E402
from src.nlp import analyze  # noqa: E402

st.set_page_config(page_title="ContagionLens", layout="wide")
st.title("ContagionLens - contagion-aware financial risk engine")
if not (C.OUT / "signals.csv").exists():
    st.error("No outputs yet. Run `python main.py all` first.")
    st.stop()

sig = pd.read_csv(C.OUT / "signals.csv", parse_dates=["timestamp"])
val = json.loads((C.OUT / "validation.json").read_text())
tabs = st.tabs(["Risk signals", "A: Index rebalancer", "B: Stress test", "C: Validation", "D: Risk brief", "Contagion graph"])

with tabs[0]:
    st.subheader("Live text analyser")
    txt = st.text_input("Paste a headline or tweet", "Boeing misses bond payment, default fears grow")
    src = st.radio("Source", ["news", "twitter"], horizontal=True)
    r = analyze(txt, src)
    r["entities"] = [f"{e} ({s})" for e, s in r["entities"]]
    st.json(r)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(sig, x="impact", color="source", nbins=20, title="Impact score distribution"), width="stretch")
    c2.plotly_chart(px.histogram(sig, x="event_type", color="source", title="Event classification"), width="stretch")
    st.dataframe(sig.sort_values("impact", ascending=False)[["timestamp", "source", "entity", "sentiment", "event_type",
                                                              "impact", "credibility", "uncorroborated", "text"]].head(200))

with tabs[1]:
    w = pd.read_csv(C.OUT / "weights.csv", index_col=0, parse_dates=True)
    st.plotly_chart(px.area(w, title="Index weights over time (contagion-aware)"), width="stretch")
    bt = pd.read_csv(C.OUT / "backtest.csv", index_col=0, parse_dates=True)
    st.plotly_chart(px.line((1 + bt).cumprod(), title="Cumulative return: strategy variants vs equal-weight"), width="stretch")
    st.dataframe(pd.DataFrame(val["backtest"]).T)

with tabs[2]:
    pf = pd.read_csv(C.DATA / "portfolio.csv")
    trig = pd.read_csv(C.OUT / "stress_triggers.csv")
    blocked = pd.read_csv(C.OUT / "stress_blocked.csv")
    st.caption(f"{len(trig)} triggered events; {len(blocked)} blocked by credibility/rumour filter")
    pick = st.selectbox("Triggered event", trig.index, format_func=lambda i: f"{trig.event_type[i]} | {trig.entity[i]} | impact {trig.impact[i]}")
    row = trig.loc[pick]
    mode = st.checkbox("Override with custom scenario")
    et, imp, ent, scope = row.event_type, float(row.impact), row.entity, row.scope
    if mode:
        et = st.selectbox("Event type", list(C.SCENARIOS))
        imp = st.slider("Impact", 1.0, 10.0, 8.0)
        ent, scope = "MARKET", "market"
    res = run_stress(pf, et, imp, ent, scope, -1.0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Before", f"{res['value_before']/1e6:,.1f}M")
    m2.metric("After", f"{res['value_after']/1e6:,.1f}M", f"{res['pnl_pct']:+.2f}%")
    m3.metric("P&L", f"{res['pnl']/1e6:,.1f}M")
    fig = go.Figure(go.Bar(x=["Before", "After"], y=[res["value_before"], res["value_after"]]))
    fig.update_yaxes(range=[min(res["value_after"], res["value_before"]) * 0.95, res["value_before"] * 1.02])
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(pd.Series(res["by_asset_type"], name="pnl").reset_index(), x="index", y="pnl", title="P&L by asset type"), width="stretch")
    b.plotly_chart(px.bar(pd.Series(res["by_sector"], name="pnl").reset_index(), x="index", y="pnl", title="P&L by sector"), width="stretch")
    st.subheader("Blocked (rumour filter)")
    st.dataframe(blocked[["date", "event_type", "entity", "impact", "credibility", "reason", "text"]].head(30))

with tabs[3]:
    st.subheader("Predictive power (rank IC vs next-day return)")
    st.dataframe(pd.DataFrame(val["ic"]).T)
    st.subheader("NLP quality vs naive baseline")
    st.dataframe(pd.DataFrame({"engine": val["nlp_quality"]["all"], "naive": val["nlp_quality_naive"]["all"]}))
    st.warning("Synthetic data: prices contain a planted event effect and text was templated, so these numbers verify "
               "the plumbing only. Re-run on real labelled text and real prices before claiming performance.")

with tabs[4]:
    st.markdown((C.OUT / "brief.md").read_text())

with tabs[5]:
    ed = pd.read_csv(C.OUT / "contagion_edges.csv")
    import math
    pos = {t: (math.cos(2 * math.pi * i / len(C.TICKERS)), math.sin(2 * math.pi * i / len(C.TICKERS))) for i, t in enumerate(C.TICKERS)}
    fig = go.Figure()
    for r in ed[ed.weight >= 0.2].itertuples():
        fig.add_trace(go.Scatter(x=[pos[r.source][0], pos[r.target][0]], y=[pos[r.source][1], pos[r.target][1]],
                                 mode="lines", line=dict(width=r.weight * 6, color="rgba(120,120,120,.4)"), showlegend=False))
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode="markers+text",
                             text=list(pos), textposition="top center", marker=dict(size=18), showlegend=False))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch")
