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

# Configure page metadata
st.set_page_config(
    page_title="S&P Global & CRISIL | Enterprise Risk Intelligence",
    page_icon="https://www.spglobal.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Institutional CSS (Inter typography, white/blue corporate palette)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --navy: #102A43;
    --blue: #1769AA;
    --blue-dark: #125585;
    --line: #D9E2EC;
    --line-strong: #B8C7D9;
    --text: #243B53;
    --muted: #627D98;
    --bg: #F4F7FB;
    --pad: 2.5rem;
    --nav-offset: 6px;
    --toggle-nudge: -2px;
}

/* ---------- Base ---------- */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    background: var(--bg);
    color: var(--text);
}
[data-testid="stAppViewContainer"] { background: transparent; }

.stApp button, .stApp input, .stApp textarea, .stApp p, .stApp label, .stApp li,
.stApp summary, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-baseweb="select"] div, [data-baseweb="popover"] li {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
code, pre, .numeric-val, .metric-number { font-family: 'IBM Plex Mono', monospace !important; }

/* The container holding this <style> tag is an empty flex item that adds a 1rem gap above the
   header (flex gap). Remove it from layout; the CSS itself keeps applying. */
[data-testid="stElementContainer"]:has(style) { display: none !important; }

/* No Streamlit chrome: navigation lives in the top bar. The sidebar (watchlist) is fixed open. */
[data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }

/* Page container: full width, flush with the top edge */
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 100% !important;
    padding-left: var(--pad) !important;
    padding-right: var(--pad) !important;
    padding-bottom: 3rem !important;
}

/* ---------- Typography scale ---------- */
[data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5, [data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] h1 *, [data-testid="stAppViewContainer"] h2 *,
[data-testid="stAppViewContainer"] h3 *, [data-testid="stAppViewContainer"] h4 *,
[data-testid="stAppViewContainer"] h5 *, [data-testid="stAppViewContainer"] h6 * {
    color: var(--navy) !important;
}
[data-testid="stAppViewContainer"] h3 { font-size: 22px !important; font-weight: 700 !important; padding: 0.4rem 0 0.2rem 0 !important; letter-spacing: -0.2px; }
[data-testid="stAppViewContainer"] h4 { font-size: 17px !important; font-weight: 600 !important; }
[data-testid="stAppViewContainer"] h5 { font-size: 15px !important; font-weight: 600 !important; }
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li { font-size: 14px; color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 13px !important; color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] label p {
    font-size: 13px !important; font-weight: 500 !important; color: var(--text) !important;
}
[data-testid="stAlert"] p { color: inherit !important; }

/* ---------- App bar: brand header + navigation row (one seamless unit) ---------- */
.st-key-appbar {
    width: calc(100% + 2 * var(--pad)) !important;
    max-width: none !important;
    margin-left: calc(-1 * var(--pad)) !important;
    margin-right: calc(-1 * var(--pad)) !important;
    margin-bottom: 1.5rem !important;
    gap: 0 !important;
    box-shadow: 0 4px 14px rgba(16, 42, 67, 0.08);
}
.st-key-appbar [data-testid="stElementContainer"] { margin: 0 !important; }
.sp-header {
    background: var(--navy);
    box-sizing: border-box;
    min-height: 84px;
    padding: 16px var(--pad);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sp-header--auth { padding-right: calc(var(--pad) + 132px); }
.sp-logo-title { color: #FFFFFF !important; font-size: 18px; font-weight: 700; letter-spacing: -0.2px; }
.sp-logo-subtitle { color: #9FB3C8 !important; font-size: 11px; font-weight: 500; letter-spacing: 0.6px; text-transform: uppercase; margin-top: 2px; }
.user-pill {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    padding: 7px 14px;
}
.user-pill-name { color: #FFFFFF; font-size: 13px; font-weight: 600; }
.user-pill-meta { color: #BCCCDC; font-size: 11px; margin-top: 1px; }

.st-key-navbar {
    background: #FFFFFF;
    border-bottom: 1px solid var(--line);
    padding: 14px var(--pad) !important;
}

/* Navigation: keyed buttons rendered as pill tabs (same 40px height as the controls) */
.st-key-navbar [data-testid="stHorizontalBlock"] {
    align-items: flex-start !important; gap: 0.6rem !important; margin: 0 !important; padding: 0 !important;
    transform: translateY(var(--nav-offset));
}
.st-key-navbar [data-testid="stColumn"] { margin: 0 !important; padding: 0 !important; }
.st-key-navbar [class*="st-key-navtab_"] button[data-testid^="stBaseButton-"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E2EC !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 10px !important;
}
.st-key-navbar [class*="st-key-navtab_"] button p {
    color: var(--muted) !important; font-size: 14px !important; font-weight: 600 !important; white-space: nowrap;
}
.st-key-navbar [class*="st-key-navtab_"] button:hover { background: #F0F6FA !important; border-color: #B8C7D9 !important; }
.st-key-navbar [class*="st-key-navtab_"] button:hover p { color: var(--navy) !important; }
.st-key-navbar [class*="st-key-navtab_active_"] button[data-testid^="stBaseButton-"] {
    background: #E8F1F8 !important; border-color: #9DC3E6 !important;
}
.st-key-navbar [class*="st-key-navtab_active_"] button p { color: var(--blue) !important; }

/* Sign Out: top-right corner of the header, in the spot the login page shows "Not signed in" */
.st-key-appbar { position: relative; }
.st-key-appbar .st-key-signout_btn {
    position: absolute !important;
    top: 22px;
    right: var(--pad);
    width: auto !important;
    margin: 0 !important;
    z-index: 20;
}
.st-key-appbar .st-key-signout_btn button[data-testid^="stBaseButton-"] {
    min-width: 112px;
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    box-shadow: none !important;
}
.st-key-appbar .st-key-signout_btn button p { color: #FFFFFF !important; }
.st-key-appbar .st-key-signout_btn button:hover {
    background: #B91C1C !important; border-color: #B91C1C !important; box-shadow: none !important;
}
.st-key-appbar .st-key-signout_btn button:hover p { color: #FFFFFF !important; }

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input {
    color: #172033 !important; background: #FFFFFF !important; font-size: 14px !important;
}
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color: #829AB1 !important; }

/* Selectbox: one clean white control with a CSS chevron */
[data-baseweb="select"] svg { display: none !important; }
[data-baseweb="select"] > div {
    position: relative !important;
    background-color: #FFFFFF !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 6px !important;
    min-height: 40px !important;
    padding-right: 38px !important;
}
[data-baseweb="select"] > div::after {
    content: ""; position: absolute; right: 15px; top: 50%;
    width: 7px; height: 7px;
    border-right: 2px solid var(--navy); border-bottom: 2px solid var(--navy);
    transform: translateY(-70%) rotate(45deg); pointer-events: none;
}
[data-baseweb="select"] > div > div, [data-baseweb="select"] > div > div > div { background: transparent !important; }
[data-baseweb="select"] > div > div:first-child { padding-left: 12px !important; }
[data-baseweb="select"] div { font-size: 13px !important; font-weight: 500 !important; color: #172033 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div { background: #FFFFFF !important; }
[data-baseweb="popover"] { border: 1px solid var(--line-strong) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: #172033 !important; background: transparent; }
[data-baseweb="popover"] li:hover { background: #F0F6FA !important; }

/* ---------- Buttons: same height / type everywhere ---------- */
button[data-testid^="stBaseButton-"] {
    height: 40px; min-height: 40px;
    border-radius: 6px;
    font-size: 13px; font-weight: 600;
    transition: all 0.15s ease;
}
button[data-testid^="stBaseButton-"] p { font-size: 13px !important; font-weight: 600 !important; margin: 0 !important; }

button[data-testid^="stBaseButton-secondary"] {
    background: #FFFFFF !important; border: 1px solid var(--line-strong) !important; color: var(--navy) !important;
}
button[data-testid^="stBaseButton-secondary"] p { color: var(--navy) !important; }
button[data-testid^="stBaseButton-secondary"]:hover { border-color: var(--blue) !important; box-shadow: 0 3px 10px rgba(23, 105, 170, 0.16); }
button[data-testid^="stBaseButton-secondary"]:hover p { color: var(--blue) !important; }

button[data-testid^="stBaseButton-primary"] {
    background: var(--blue) !important; border: 1px solid var(--blue) !important; color: #FFFFFF !important;
}
button[data-testid^="stBaseButton-primary"] p { color: #FFFFFF !important; }
button[data-testid^="stBaseButton-primary"]:hover { background: var(--blue-dark) !important; border-color: var(--blue-dark) !important; box-shadow: 0 3px 10px rgba(23, 105, 170, 0.28); }

/* ---------- Login card ---------- */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-top: 4px solid var(--blue);
    border-radius: 10px;
    padding: 38px 42px 34px 42px;
    box-shadow: 0 12px 30px rgba(16, 42, 67, 0.08);
}

/* ---------- Metric cards ---------- */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 22px; }
.metric-box { background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(16, 42, 67, 0.04); }
.metric-title { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
.metric-number { font-size: 22px; font-weight: 700; color: var(--navy); }
.metric-footnote { font-size: 11px; color: #64748B; margin-top: 4px; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
[data-testid="stMetricValue"] { font-size: 22px !important; }
[data-testid="stMetricLabel"] p { font-size: 12px !important; }

/* ---------- Record cards & tags ---------- */
.record-card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(16, 42, 67, 0.035); }
.record-card:hover { border-color: #8FB8D8; box-shadow: 0 5px 16px rgba(16, 42, 67, 0.08); }
.tag-category, .tag-critical, .tag-high, .tag-moderate, .tag-low {
    font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;
}
.tag-category { background: #E8F1F8; color: var(--blue); border: 1px solid #B8D4EA; }
.tag-critical { background: #FDECEC; color: #B91C1C; border: 1px solid #F5C2C2; }
.tag-high     { background: #FFF1E6; color: #C2410C; border: 1px solid #F9CFAE; }
.tag-moderate { background: #FEF7E0; color: #A16207; border: 1px solid #F3DD9A; }
.tag-low      { background: #E8F7F1; color: #047857; border: 1px solid #A7E3CB; }

/* ---------- Tables, expanders, tabs ---------- */
div[data-testid="stDataFrame"] { border: 1px solid var(--line) !important; border-radius: 8px !important; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary p { font-size: 14px !important; color: var(--text) !important; }
div[data-baseweb="tab-list"] { gap: 8px; }
button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--blue); }
/* ---------- Login ---------- */
.login-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--blue); margin-bottom: 8px; }
.login-title { font-size: 26px; font-weight: 700; color: var(--navy); letter-spacing: -0.4px; line-height: 1.2; margin-bottom: 8px; }
.login-sub { font-size: 13px; color: var(--muted); line-height: 1.55; margin-bottom: 24px; }
.login-foot { text-align: center; font-size: 12px; color: #829AB1; margin-top: 18px; }
[data-testid="stForm"] [data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 600 !important; }
[data-testid="stForm"] [data-baseweb="input"] { min-height: 46px !important; }
[data-testid="stForm"] input { font-size: 15px !important; padding-left: 14px !important; }
[data-testid="stForm"] button[data-testid^="stBaseButton-"] { height: 48px; min-height: 48px; margin-top: 6px; }
[data-testid="stForm"] button[data-testid^="stBaseButton-"] p { font-size: 15px !important; }

/* ---------- News record: card + its analysis are ONE unit ---------- */
[class*="st-key-rec_"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px 20px 4px 20px;
    box-shadow: 0 2px 8px rgba(16, 42, 67, 0.035);
    gap: 0 !important;
}
[class*="st-key-rec_"]:hover { border-color: #8FB8D8; box-shadow: 0 5px 16px rgba(16, 42, 67, 0.08); }
[class*="st-key-rec_"] .record-card { background: transparent; border: none; box-shadow: none; padding: 0; margin: 0; }
[class*="st-key-rec_"] [data-testid="stExpander"],
[class*="st-key-rec_"] [data-testid="stExpander"] details,
[class*="st-key-rec_"] [data-testid="stExpander"] summary,
[class*="st-key-rec_"] [data-testid="stExpander"] > div {
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    outline: none !important;
}
[class*="st-key-rec_"] [data-testid="stExpander"] { margin-top: 6px; }
[class*="st-key-rec_"] [data-testid="stExpander"] summary { padding: 8px 0 !important; }
[class*="st-key-rec_"] [data-testid="stExpander"] summary:hover { background: transparent !important; }
[class*="st-key-rec_"] [data-testid="stExpander"] summary p { font-size: 13px !important; font-weight: 600 !important; color: var(--blue) !important; }

/* ---------- Fetch & Analyze progress bar (indeterminate) ---------- */
.fx-bar { height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; margin: -0.5rem 0 1rem 0; }
.fx-bar::before { content: ""; display: block; height: 100%; width: 32%; background: var(--blue); border-radius: 2px; animation: fx-slide 1.1s ease-in-out infinite; }
@keyframes fx-slide { 0% { margin-left: -32%; } 100% { margin-left: 100%; } }

/* ---------- Governance footer (a standing statement, not a toggle) ---------- */
.gov-panel { border-top: 1px solid var(--line); margin-top: 48px; padding-top: 18px; }
.gov-grid { display: grid; grid-template-columns: 1.1fr 1.5fr 1.5fr; gap: 36px; align-items: start; }
.gov-title { font-size: 11px; font-weight: 700; letter-spacing: 0.9px; text-transform: uppercase; color: var(--muted); }
.gov-sub { font-size: 12px; color: var(--muted); margin: 6px 0 0 0; line-height: 1.5; }
.gov-badge { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: 0.6px; color: var(--blue); }
.gov-item-title { font-size: 12.5px; font-weight: 600; color: var(--navy); margin: 4px 0 4px 0; }
.gov-item-text { font-size: 12px; color: var(--muted); line-height: 1.6; }
@media (max-width: 900px) { .gov-grid { grid-template-columns: 1fr; gap: 18px; } }
@media (min-width: 1100px) { .login-sub { white-space: nowrap; } }

/* ---------- Clearly boxed text inputs ---------- */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="base-input"] {
    border: none !important; box-shadow: none !important; background: transparent !important; border-radius: 8px !important;
}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"] {
    border: 1.5px solid #AFC0D2 !important;
    border-radius: 8px !important;
    background: #F8FAFC !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within {
    border-color: var(--blue) !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(23, 105, 170, 0.15) !important;
}
[data-testid="stTextInput"] input { background: transparent !important; }
[data-testid="stForm"] [data-testid="stTextInputRootElement"] { min-height: 46px !important; }
/* ---------- Demo-data toggle: text is a pseudo-element, so only the switch is clickable ---------- */
.st-key-demo_wrap { height: 40px; flex-direction: row !important; align-items: center !important; justify-content: flex-end; gap: 10px !important; }
.st-key-demo_wrap > [data-testid="stElementContainer"] { width: auto !important; }
.demo-label { display: flex; align-items: center; gap: 7px; height: 40px; line-height: 1; margin: 0 !important; padding: 0 !important; font-size: 13px; font-weight: 600; color: var(--muted); white-space: nowrap; }
.st-key-demo_wrap [data-testid="stMarkdown"],
.st-key-demo_wrap [data-testid="stMarkdownContainer"] { margin: 0 !important; padding: 0 !important; display: flex; align-items: center; height: 40px; }
.st-key-demo_wrap [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
.st-key-demo_wrap > [data-testid="stElementContainer"] { display: flex; align-items: center; height: 40px; }
.st-key-demo_toggle { height: 40px; display: flex !important; align-items: center; }
.st-key-demo_toggle [data-testid="stWidgetLabel"] { display: none !important; }
.st-key-demo_toggle [data-testid="stCheckbox"] {
    margin: 0 !important; padding: 0 !important; min-height: 0 !important;
    display: flex; align-items: center;
    position: relative; top: var(--toggle-nudge);
}
.st-key-demo_toggle label { margin: 0 !important; padding: 0 !important; min-height: 0 !important; align-items: center !important; }

/* ---------- Left rail: filters + watchlist (tinted so it reads as a control panel, not a KPI card) ---------- */
.st-key-wl_panel {
    background: #EAF0F7; border: 1px solid #D3DEEA; border-radius: 10px;
    padding: 16px 14px 14px 14px; gap: 0.5rem !important;
}
.wl-title { font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin: 0; padding: 4px 0 14px 0; line-height: 1.2; }
.st-key-wl_panel [data-testid="stElementContainer"]:has(.wl-title) { margin-bottom: 6px !important; }
.wl-sep { border-top: 1px solid #D3DEEA; margin: 12px 0 8px 0; }
.st-key-wl_panel [data-testid="stWidgetLabel"] p { font-size: 12px !important; font-weight: 600 !important; color: var(--text) !important; }
.st-key-wl_panel [data-testid="stSliderThumbValue"], .st-key-wl_panel [data-testid="stSliderTickBarMin"], .st-key-wl_panel [data-testid="stSliderTickBarMax"] { font-size: 12px !important; }
.st-key-wl_panel [data-testid="stCheckbox"] label p, .st-key-wl_panel [data-testid="stToggle"] label p { font-size: 13px !important; }
.st-key-wl_panel [data-testid="stElementContainer"] { margin: 0 !important; }

/* watchlist rows: one small white card per ticker */
[class*="st-key-wlrow_"] { background: #FFFFFF; border: 1px solid #DCE5EF; border-radius: 8px; padding: 6px 10px 12px 10px; gap: 0 !important; overflow: visible; }
[class*="st-key-wlrow_"] [data-testid="stMarkdown"], [class*="st-key-wlrow_"] [data-testid="stMarkdownContainer"] { margin: 0 !important; padding: 0 !important; }
[class*="st-key-wlrow_"] [data-testid="stHorizontalBlock"] { gap: 0.25rem !important; align-items: center; }
[class*="st-key-wlrow_"] button[data-testid^="stBaseButton-"] { min-height: 28px !important; padding: 0 !important; }
[class*="st-key-wlrow_"] [class*="st-key-wlf_"] button { justify-content: flex-start !important; }
[class*="st-key-wlrow_"] [class*="st-key-wlf_"] button > div, [class*="st-key-wlrow_"] [class*="st-key-wlf_"] button p { width: 100%; text-align: left !important; justify-content: flex-start !important; }
[class*="st-key-wlrow_"] [class*="st-key-wlf_"] button p { font-size: 14px !important; font-weight: 700 !important; letter-spacing: 0.2px; }
[class*="st-key-wlrow_"] [class*="st-key-wlx_"] button { color: var(--muted) !important; }
[class*="st-key-wlrow_"] [class*="st-key-wlx_"] button:hover { color: #B91C1C !important; background: transparent !important; }
.wl-meta { font-size: 11.5px; line-height: 1.4; color: var(--muted); margin: 0; padding: 0 0 2px 0; }
.wl-meta b { font-weight: 600; }

/* Watchlist checkbox: same 40px height as the select box beside it, vertically centred */
.st-key-watchlist_only { height: 40px; display: flex !important; align-items: center; }
.st-key-watchlist_only [data-testid="stCheckbox"] { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }

/* ---------- Watchlist drawer: overlays the page from the left, never shifts content ---------- */
[data-testid="stLayoutWrapper"]:has(> .st-key-wl_drawer) { display: contents !important; }
.st-key-wl_drawer {
    position: fixed !important;
    top: 0; left: 0;
    width: 340px; max-width: 92vw; height: 100vh;
    box-sizing: border-box;
    overflow-y: auto;
    background: #EAF0F7;
    border-right: 1px solid #D3DEEA;
    box-shadow: 10px 0 34px rgba(16, 42, 67, 0.22);
    padding: 18px 16px 24px 16px;
    gap: 0.5rem !important;
    z-index: 1000;
    animation: wl-slide-in 0.18s ease-out;
}
[data-testid="stLayoutWrapper"]:has(> .st-key-wl_handle) { display: contents !important; }
.st-key-wl_handle { position: absolute !important; left: 0; top: 144px; width: 38px !important; z-index: 30; margin: 0 !important; }
.st-key-wl_handle button[data-testid^="stBaseButton-"] {
    width: 38px !important; min-width: 0 !important; padding: 0 !important;
    border-radius: 0 8px 8px 0 !important; border-left: none !important;
    box-shadow: 3px 3px 10px rgba(16, 42, 67, 0.25) !important;
}
.st-key-wl_handle button p { font-size: 16px !important; }
.st-key-wl_drawer [data-testid="stElementContainer"] { margin: 0 !important; }
.st-key-wl_drawer .wl-title { padding-bottom: 0; }
.st-key-wl_drawer [class*="st-key-wl_close"] button { color: var(--muted) !important; }
@keyframes wl-slide-in { from { transform: translateX(-100%); } to { transform: translateX(0); } }

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# RESOURCE CACHING (Clean, Silent, Non-Intrusive)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_risk_engine():
    return NLPFinancialRiskEngine(enable_transformer=True, enable_ner=True)

@st.cache_resource(show_spinner=False)
def load_stress_engine():
    return WholesalePortfolioStressEngine()

@st.cache_resource(show_spinner=False)
def load_rebalancer():
    return TacticalIndexRebalancer()

@st.cache_resource(show_spinner=False)
def load_security_master():
    with open("data/reference/security_master.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_user_registry():
    with open("data/demo/users_profiles.json", "r", encoding="utf-8") as f:
        return json.load(f)["users"]

engine = load_risk_engine()
stress_engine = load_stress_engine()
rebalancer = load_rebalancer()
security_master = load_security_master()
users_list = load_user_registry()
store = RiskSignalStore()

# -------------------------------------------------------------
# SESSION STATE & AUTHENTICATION MANAGEMENT
# -------------------------------------------------------------
if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

def render_header(name_line, meta_line, authed=False):
    """Brand bar shared by the login page and the main console."""
    header_class = "sp-header sp-header--auth" if authed else "sp-header"
    st.markdown(f"""<div class="{header_class}">
    <div>
        <div class="sp-logo-title">S&P Global & CRISIL</div>
        <div class="sp-logo-subtitle">Financial Risk Intelligence & Portfolio Stress Testing Engine</div>
    </div>
    <div class="user-pill">
        <div class="user-pill-name">{name_line}</div>
        <div class="user-pill-meta">{meta_line}</div>
    </div>
</div>""", unsafe_allow_html=True)


# If not authenticated, render the login page inside the same app shell
if st.session_state["authenticated_user"] is None:
    with st.container(key="appbar"):
        render_header("Not signed in", "Secure Desk Access")

    st.markdown("<div style='height: 56px;'></div>", unsafe_allow_html=True)

    _, login_mid, _ = st.columns([0.85, 1.9, 0.85])
    with login_mid:
        with st.form("auth_form"):
            st.markdown("""<div class="login-eyebrow">Secure Desk Access</div>
<div class="login-title">Sign in to the Risk Console</div>
<div class="login-sub">Use your corporate credentials to access market risk, rebalancing and stress testing.</div>""", unsafe_allow_html=True)
            username_input = st.text_input("Corporate ID / Username", placeholder="e.g. sarah.chen, marcus.vance, or admin")
            password_input = st.text_input("Password", type="password", placeholder="Enter your credentials")
            submit_login = st.form_submit_button("Access Risk Console", width="stretch", type="primary")

            if submit_login:
                matched = next(
                    (u for u in users_list
                     if u["username"] == username_input.strip()
                     and u.get("password") == password_input.strip()),
                    None,
                )
                if matched:
                    st.session_state["authenticated_user"] = {**matched, "watchlist": list(matched.get("watchlist", []))}
                    st.rerun()
                else:
                    st.error("Authentication failed. Please verify credentials.")

        st.markdown('<div class="login-foot">Authorized use only &bull; Access is monitored and logged</div>', unsafe_allow_html=True)

    st.stop()

# -------------------------------------------------------------
# AUTHENTICATED PORTAL WORKSPACE
# -------------------------------------------------------------
current_user = st.session_state["authenticated_user"]
is_admin = current_user["user_id"] == "usr_risk_admin"

nav_options = [
    "1. Market Signals & News Intelligence",
    "2. Dynamic Index Allocation & Rebalancing",
    "3. Balance Sheet Stress Testing",
]
if is_admin:
    nav_options.append("4. Securities Master & Coverage Directory")

nav_labels = {
    "1. Market Signals & News Intelligence": "1. Market Signals",
    "2. Dynamic Index Allocation & Rebalancing": "2. Index Rebalancing",
    "3. Balance Sheet Stress Testing": "3. Stress Testing",
    "4. Securities Master & Coverage Directory": "4. Securities Master",
}
feed_channels = ["Demo Market Scenarios", "Live Market News (CNBC / Yahoo)"]

def set_workspace(option):
    st.session_state["workspace"] = option


def _wl_bump():
    st.session_state["wl_ver"] = st.session_state.get("wl_ver", 0) + 1


def _wl_toggle():
    st.session_state["wl_open"] = not st.session_state.get("wl_open", False)


def _wl_close():
    st.session_state["wl_open"] = False


if st.session_state.get("workspace") not in nav_options:
    st.session_state["workspace"] = nav_options[0]
selected_workspace = st.session_state["workspace"]

# App bar: brand header + navigation + ingestion controls (replaces the sidebar)
with st.container(key="appbar"):
    render_header(
        current_user["display_name"],
        f"{current_user['role']} &bull; {current_user['desk']}",
        authed=True,
    )
    # Sign Out lives in the header's top-right corner (positioned by CSS)
    sign_out_clicked = st.button("Sign Out", key="signout_btn")
    wl_holder = st.container(key="wl_holder")  # watchlist handle/drawer render here (filled further down)
    with st.container(key="navbar"):
        tab_widths = [1.75] * len(nav_options)
        wl_slot = 0
        wl_btn_width = 1.5
        spacer_width = (3.9 if len(nav_options) == 3 else 2.0) - (wl_btn_width if wl_slot else 0)
        bar_cols = st.columns(
            tab_widths + ([wl_btn_width] if wl_slot else []) + [spacer_width, 1.45, 1.25],
            gap="small",
        )
        for i, option in enumerate(nav_options):
            with bar_cols[i]:
                is_active = option == selected_workspace
                st.button(
                    nav_labels[option],
                    key=f"navtab_{'active_' if is_active else ''}{i}",
                    on_click=set_workspace,
                    args=(option,),
                    width="stretch",
                )
        n = len(nav_options)
        with bar_cols[n + wl_slot + 1]:
            fetch_clicked = st.button(
                "Fetch & Analyze",
                type="primary",
                width="stretch",
            )
        with bar_cols[n + wl_slot + 2]:
            # The label is plain HTML; only the switch itself is clickable.
            # On = deterministic scenarios, Off = live financial wire.
            with st.container(key="demo_wrap"):
                st.markdown(
                    '<div class="demo-label" title="On: use prepared sample market scenarios (for demos). '
                    'Off: pull live market news (CNBC / Yahoo).">Demo data</div>',
                    unsafe_allow_html=True,
                )
                demo_mode = st.toggle(
                    "Demo data",
                    value=True,
                    key="demo_toggle",
                    label_visibility="collapsed",
                )
            feed_selection = feed_channels[0] if demo_mode else feed_channels[1]

if sign_out_clicked:
    st.session_state["authenticated_user"] = None
    st.session_state.pop("wl_edit", None)
    st.session_state.pop("wl_focus", None)
    st.session_state.pop("wl_open", None)
    st.rerun()

if fetch_clicked:
    # Visual progress instead of status text: a slim bar directly under the nav row
    _fx = st.empty()
    _fx.markdown('<div class="fx-bar"></div>', unsafe_allow_html=True)
    try:
        adapter = ReplayAdapter() if feed_selection == feed_channels[0] else RSSFeedAdapter()
        articles = adapter.fetch_articles()
        processed_signals = engine.process_batch(articles)
    finally:
        _fx.empty()

signals = store.get_all_signals(limit=100)
st.session_state["wl_ver_seen"] = st.session_state.get("wl_ver", 0)

# -------------------------------------------------------------
# WATCHLIST SIDEBAR (personal preference: session-only add/remove)
# -------------------------------------------------------------
if not is_admin:
    if "wl_edit" not in st.session_state:
        st.session_state["wl_edit"] = list(current_user.get("watchlist", []))
    _names = {m["ticker"]: m["name"] for m in security_master}

    def _wl_add():
        t = st.session_state.get("wl_add")
        if t and t not in st.session_state["wl_edit"]:
            st.session_state["wl_edit"].append(t)
        st.session_state["wl_add"] = None
        _wl_bump()

    def _wl_remove(t):
        st.session_state["wl_edit"] = [x for x in st.session_state["wl_edit"] if x != t]
        if st.session_state.get("wl_focus") == t:
            st.session_state["wl_focus"] = None
        _wl_bump()

    def _wl_focus(t):
        st.session_state["wl_focus"] = None if st.session_state.get("wl_focus") == t else t
        _wl_bump()

    current_user["watchlist"] = list(st.session_state["wl_edit"])
    _wl = current_user["watchlist"]
    _focus = st.session_state.get("wl_focus")

    def _render_watchlist():
        c_t, c_c = st.columns([5, 1], vertical_alignment="center", gap="small")
        with c_t:
            st.markdown(f'<div class="wl-title">\u2605 Watchlist &bull; {len(_wl)}</div>', unsafe_allow_html=True)
        with c_c:
            st.button("\u2715", key="wl_close", on_click=_wl_close, type="tertiary", help="Close watchlist")
        st.caption("Click a ticker to focus the Market Signals feed on it.")
        st.selectbox(
            "Add ticker",
            options=[t for t in sorted(_names) if t not in _wl],
            index=None,
            placeholder="+ Add ticker",
            format_func=lambda t: f"{t} \u2014 {_names[t]}",
            key="wl_add",
            on_change=_wl_add,
            label_visibility="collapsed",
        )
        if not _wl:
            st.caption("No tickers yet.")
        for t in _wl:
            _ts = [s for s in signals if s.ticker == t]
            with st.container(key=f"wlrow_{t}"):
                c_main, c_x = st.columns([6, 1], vertical_alignment="center", gap="small")
                with c_main:
                    st.button(
                        t, key=f"wlf_{t}", on_click=_wl_focus, args=(t,), help=_names.get(t, t),
                        type="primary" if _focus == t else "tertiary", width="stretch",
                    )
                with c_x:
                    st.button("\u2715", key=f"wlx_{t}", on_click=_wl_remove, args=(t,), type="tertiary", help=f"Remove {t}")
                if _ts:
                    _avg = float(np.mean([x.sentiment_score for x in _ts]))
                    _col = "#10B981" if _avg > 0.15 else ("#EF4444" if _avg < -0.15 else "#486581")
                    _mx = max(x.impact_score for x in _ts)
                    _mx_col = "#EF4444" if _mx >= 7.0 else "inherit"
                    st.markdown(
                        f'<div class="wl-meta"><b style="color:{_col}">{_avg:+.2f}</b> &bull; {len(_ts)} sig &bull; '
                        f'max <b style="color:{_mx_col}">{_mx:.1f}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="wl-meta">No signals</div>', unsafe_allow_html=True)

    @st.fragment
    def _watchlist_fragment():
        """Handle + drawer. Open/close reruns only this fragment; edits trigger a full rerun."""
        if st.session_state.get("wl_ver", 0) != st.session_state.get("wl_ver_seen", 0):
            st.rerun()
        if st.session_state.get("wl_open"):
            with st.container(key="wl_drawer"):
                _render_watchlist()
        else:
            st.button(
                "\u2605",
                key="wl_handle", on_click=_wl_toggle, type="primary",
                help=f"Open watchlist ({len(st.session_state['wl_edit'])})",
            )

    with wl_holder:
        _watchlist_fragment()

# The stress engine values swap baselines at 2% of notional, so use the same
# convention when displaying the consolidated portfolio label.
consolidated_baseline = sum(
    position.get(
        "market_value",
        position.get(
            "ead",
            position.get("notional", 0.0) * 0.02
            if position["asset_type"] == "Swap"
            else 0.0,
        ),
    )
    for position in stress_engine.positions
)
consolidated_label = (
    f"Consolidated Wholesale Banking Portfolio "
    f"(${consolidated_baseline / 1e6:.1f}M Bank-Wide)"
)

# -------------------------------------------------------------
# WORKSPACE 1: Market Signals & Unstructured Ingestion
# -------------------------------------------------------------
if selected_workspace == "1. Market Signals & News Intelligence":
    st.markdown("### Market Risk Intelligence")
    st.caption("Continuously reviews financial news, regulatory actions and market updates, and turns each story into a sentiment score, an event type and a 10-point severity rating.")

    # KPI Metric Row
    critical_signals_count = sum(1 for s in signals if s.impact_score >= 7.0)
    avg_market_sentiment = np.mean([s.sentiment_score for s in signals]) if signals else 0.0
    active_entity_count = len(set(s.ticker for s in signals if s.ticker))

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-title">Stories Reviewed</div>
            <div class="metric-number">{len(signals)}</div>
            <div class="metric-footnote">Unique Stories, Duplicates Removed</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Material Events Requiring Review</div>
            <div class="metric-number" style="color:#EF4444;">{critical_signals_count}</div>
            <div class="metric-footnote">Severity Score &ge; 7.0 / 10</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Aggregated Market Sentiment</div>
            <div class="metric-number" style="color:{'#10B981' if avg_market_sentiment >= 0 else '#EF4444'};">{avg_market_sentiment:+.2f}</div>
            <div class="metric-footnote">Average Tone Across All Stories</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Portfolio Assets Impacted</div>
            <div class="metric-number">{active_entity_count}</div>
            <div class="metric-footnote">Companies Named Directly in Stories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Left rail: filters, then the personal watchlist (non-admin). Right: signal feed.
    wl_col, feed_col = st.columns([1, 4.2], gap="medium")
    with wl_col:
        with st.container(key="wl_panel"):
            st.markdown('<div class="wl-title">Filters</div>', unsafe_allow_html=True)
            min_impact = st.slider("Minimum Risk Severity", 1.0, 10.0, 1.0, 0.5)
            category_options = ["All Event Categories"] + sorted(list(set(s.event_type for s in signals)))
            selected_category = st.selectbox("Event Classification", category_options)
            sort_by = st.selectbox("Sort by", ["Severity (high to low)", "Most recent", "Sentiment (most negative)"])
            if not is_admin:
                desk_filter_only = st.toggle("Show watchlist only", value=True, key="watchlist_only")
                if st.session_state.get("wl_focus"):
                    st.caption(f"Focused on {st.session_state['wl_focus']}")
                    st.button(
                        "Clear focus", key="wl_clear", on_click=_wl_focus,
                        args=(st.session_state["wl_focus"],), width="stretch",
                    )
            else:
                desk_filter_only = False
                st.caption("Administrator profile: full institutional universe visibility active.")

    with feed_col:
        filtered_signals = [
            s for s in signals 
            if s.impact_score >= min_impact and
               (selected_category == "All Event Categories" or s.event_type == selected_category) and
               (not desk_filter_only or (s.ticker in current_user.get("watchlist", []) or not s.ticker)) and
               (not st.session_state.get("wl_focus") or s.ticker == st.session_state["wl_focus"])
        ]

        _epoch = pd.Timestamp("1970-01-01", tz="UTC")

        def _ts(sig):
            t = pd.to_datetime(getattr(sig, "published_at", None), errors="coerce", utc=True)
            return t if pd.notna(t) else _epoch

        if sort_by == "Most recent":
            filtered_signals.sort(key=lambda x: (_ts(x), x.impact_score), reverse=True)
        elif sort_by == "Sentiment (most negative)":
            filtered_signals.sort(key=lambda x: (x.sentiment_score, -x.impact_score))
        else:
            filtered_signals.sort(key=lambda x: (x.impact_score, _ts(x)), reverse=True)

        st.markdown(f"**Discovered {len(filtered_signals)} Market Records**")

        if not filtered_signals:
            if not signals:
                st.info("No market records yet. Use 'Fetch & Analyze' in the top bar to load market data.")
            else:
                _active = []
                if min_impact > 1.0:
                    _active.append(f"severity \u2265 {min_impact:.1f}")
                if selected_category != "All Event Categories":
                    _active.append(f"category: {selected_category}")
                if desk_filter_only:
                    _active.append("watchlist only")
                if st.session_state.get("wl_focus"):
                    _active.append(f"focused on {st.session_state['wl_focus']}")
                st.info(
                    f"None of the {len(signals)} loaded records match the current filters"
                    + (f" ({'; '.join(_active)})" if _active else "")
                    + ". Try lowering the severity threshold or widening the filters."
                )
        else:
            for s in filtered_signals:
                tag_class = {
                    "CRITICAL": "tag-critical",
                    "HIGH": "tag-high",
                    "MEDIUM": "tag-moderate",
                    "LOW": "tag-low"
                }.get(s.materiality, "tag-low")

                sent_display_color = "#10B981" if s.sentiment_score > 0.15 else ("#EF4444" if s.sentiment_score < -0.15 else "#486581")

                rec_key = "rec_" + "".join(ch if ch.isalnum() else "_" for ch in str(s.signal_id))
                with st.container(key=rec_key):
                    st.markdown(f"""
                    <div class="record-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <div>
                                <span class="tag-category">{s.event_type}</span>
                                <span class="{tag_class}" style="margin-left:8px;">Impact: {s.impact_score:.1f} / 10 &bull; {s.materiality} Severity</span>
                            </div>
                            <div style="font-size:11px; color:#64748B;">Signal ID: <span class="numeric-val">{s.signal_id}</span></div>
                        </div>
                        <div style="font-size:15px; font-weight:600; color:#102A43; margin-bottom:6px;">{s.title}</div>
                        <div style="font-size:12px; color:#486581; display:flex; gap:20px;">
                            <span>Source: <strong>{s.source_name}</strong></span>
                            <span>Affected Company: <strong>{f'{s.company_name} ({s.ticker})' if s.ticker else 'Broad Market / Macro'}</strong></span>
                            <span>Timestamp: <strong>{s.published_at}</strong></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("View analytical breakdown & verifiable source evidence", expanded=False):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            st.markdown("**Verifiable Text Evidence Extracted:**")
                            st.info(f"\"{s.evidence_text}\"")
                            st.markdown("**Risk Score Drivers:**")
                            for d in s.impact_drivers:
                                st.write(f"• {d}")
                        with c2:
                            st.markdown("**Sentiment Breakdown:**")
                            st.markdown(f"<span style='font-size:20px; font-weight:700; color:{sent_display_color};'>{s.sentiment_score:+.3f}</span> ({s.sentiment_label})", unsafe_allow_html=True)
                            p_df = pd.DataFrame({
                                "Sentiment": ["Positive", "Neutral", "Negative"],
                                "Probability": [s.sentiment_probs.get("positive", 0), s.sentiment_probs.get("neutral", 0), s.sentiment_probs.get("negative", 0)]
                            })
                            fig_p = px.bar(p_df, x="Sentiment", y="Probability", color="Sentiment",
                                           color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"},
                                           height=170)
                            fig_p.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False, yaxis_range=[0, 1], template="plotly_white")
                            st.plotly_chart(
                                fig_p,
                                width="stretch",
                                key=f"sentiment-chart-{s.signal_id}",
                            )
                        with c3:
                            st.markdown("**Model Details:**")
                            st.write(f"• Model Used: `{s.model_version}`")
                            st.write(f"• Event Confidence: **{s.event_confidence*100:.1f}%**")
                            st.write(f"• Impact Confidence: **{s.impact_confidence*100:.1f}%**")

# -------------------------------------------------------------
# WORKSPACE 2: Dynamic Index Allocation & Rebalancing
# -------------------------------------------------------------
elif selected_workspace == "2. Dynamic Index Allocation & Rebalancing":
    st.markdown("### Tactical Index Rebalancing & Dynamic Tilt")
    st.caption("Adjusts the weight of each stock in the portfolio in response to the latest market sentiment, while keeping trading activity and portfolio changes within set limits.")

    allocation_tickers = None if is_admin else current_user.get("watchlist", [])
    allocation_label = (
        "Full Institutional Universe"
        if is_admin
        else f"{current_user['display_name']}'s Watchlist"
    )
    st.info(f"Allocation Universe: **{allocation_label}**")

    col_alpha, col_summary = st.columns([1, 2])
    with col_alpha:
        alpha = st.slider("Sentiment Allocation Sensitivity", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
        st.caption("Adjusts aggressive weighting toward positive-sentiment companies relative to equal-weight baseline.")

    rebalance_result = rebalancer.rebalance(
        signals,
        alpha=alpha,
        allowed_tickers=allocation_tickers,
    )
    df_reb = pd.DataFrame([p.model_dump() for p in rebalance_result.positions])

    with col_summary:
        m1, m2, m3 = st.columns(3)
        m1.metric("Universe Constituents", len(rebalance_result.positions))
        m2.metric("Portfolio Turnover", f"{rebalance_result.total_turnover * 100:.2f}%")
        top_pos = rebalance_result.positions[0]
        m3.metric("Top Overweight Target", f"{top_pos.ticker} ({top_pos.target_weight*100:.1f}%)")

    st.markdown("---")

    # Clean Bar Chart
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=df_reb["ticker"],
        y=df_reb["initial_weight"] * 100,
        name="Equal-Weight Benchmark (8.33%)",
        marker_color="#9FB3C8"
    ))
    fig_alloc.add_trace(go.Bar(
        x=df_reb["ticker"],
        y=df_reb["target_weight"] * 100,
        name="Tactical Optimized Weight (%)",
        marker_color="#2563EB"
    ))
    fig_alloc.update_layout(
        title="Portfolio Constituent Weight Reallocation: Benchmark vs Sentiment-Adjusted Allocation",
        xaxis_title="Security Symbol",
        yaxis_title="Portfolio Weight (%)",
        barmode="group",
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_alloc, width="stretch", key="allocation-chart")

    st.markdown("##### Detailed Asset Allocation")
    disp_reb = df_reb[["ticker", "company_name", "sector", "current_sentiment", "initial_weight", "target_weight", "weight_delta"]].copy()
    disp_reb["initial_weight"] = (disp_reb["initial_weight"] * 100).round(2).astype(str) + "%"
    disp_reb["target_weight"] = (disp_reb["target_weight"] * 100).round(2).astype(str) + "%"
    disp_reb["weight_delta"] = (disp_reb["weight_delta"] * 100).map(lambda x: f"{x:+.2f}%")
    disp_reb["current_sentiment"] = disp_reb["current_sentiment"].map(lambda x: f"{x:+.3f}")

    st.dataframe(
        disp_reb.rename(columns={
            "ticker": "Symbol", "company_name": "Company", "sector": "Industry Sector",
            "current_sentiment": "Sentiment Change", "initial_weight": "Benchmark Weight",
            "target_weight": "Optimized Weight", "weight_delta": "Tactical Delta"
        }),
        width="stretch",
        hide_index=True
    )

# -------------------------------------------------------------
# WORKSPACE 3: Balance Sheet Stress Testing
# -------------------------------------------------------------
elif selected_workspace == "3. Balance Sheet Stress Testing":
    st.markdown("### Strategic Balance Sheet Stress Testing")
    st.caption("Tests whether the bank's capital holds up under severe market and credit shocks, across corporate loans, bonds, equity holdings and interest-rate contracts.")

    eligible_stress_signals = [
        s for s in signals
        if s.impact_score >= 7.0
        and s.impact_confidence >= 0.65
        and s.corroboration_count >= 2
    ]

    st.markdown("##### 1. Select Market Event Scenario")
    if not eligible_stress_signals:
        st.warning(
            "No market signals currently satisfy the formal stress policy "
            "(impact >= 7.0 and confidence >= 0.65). Use 'Fetch & Analyze' "
            "to load more market news."
        )
        chosen_signal = None
    else:
        sig_map = {
            f"[{s.event_type}] {s.title} "
            f"(Severity: {s.impact_score:.1f}/10, Confidence: {s.impact_confidence:.2f})": s
            for s in eligible_stress_signals
        }
        selected_event_key = st.selectbox("Choose a Stress Scenario:", list(sig_map.keys()))
        chosen_signal = sig_map[selected_event_key]

    assigned_book_label = (
        f"{current_user['display_name']}'s Assigned Book "
        f"({current_user['portfolio_name']})"
    )
    if is_admin:
        portfolio_options = [consolidated_label]
        test_positions = None
    else:
        portfolio_options = [assigned_book_label]
        test_positions = current_user["positions"]

    portfolio_selection = st.radio(
        "Target Portfolio Book:",
        portfolio_options,
        horizontal=True,
    )

    if chosen_signal:
        st.caption(
            f"Corroborated by {chosen_signal.corroboration_count} independent source(s): "
            f"{', '.join(chosen_signal.corroborating_sources)}"
        )
        stress_res = stress_engine.run_stress_test(
            signal_title=chosen_signal.title,
            event_type=chosen_signal.event_type,
            impact_score=chosen_signal.impact_score,
            confidence=chosen_signal.impact_confidence,
            source_name=chosen_signal.source_name,
            affected_ticker=chosen_signal.ticker,
            custom_positions=test_positions,
            corroboration_count=chosen_signal.corroboration_count,
            novelty_score=chosen_signal.novelty_score,
        )

        st.markdown("---")
        # Policy Gate Box
        g1, g2 = st.columns([1, 2])
        with g1:
            if stress_res.policy_decision.accepted:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 4px; padding: 14px;">
                    <div style="color: #10B981; font-weight: 700; font-size: 12px; text-transform: uppercase;">Policy Gate: Approved for Stress Testing</div>
                    <div style="color: #486581; font-size: 12px; margin-top: 6px;">{stress_res.policy_decision.trigger_reason}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 4px; padding: 14px;">
                    <div style="color: #EF4444; font-weight: 700; font-size: 12px; text-transform: uppercase;">Policy Gate: Filtered as Non-Material</div>
                    <div style="color: #486581; font-size: 12px; margin-top: 6px;">{stress_res.policy_decision.trigger_reason}</div>
                </div>
                """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div style="background: #F0F6FA; border: 1px solid #C7D7E5; border-radius: 4px; padding: 14px;">
                <div style="color: #1769AA; font-weight: 600; font-size: 12px; text-transform: uppercase;">Simulated Stress Parameters: {stress_res.scenario_name}</div>
                <div style="color: #486581; font-size: 12px; margin-top: 6px;">
                    {' &bull; '.join(stress_res.assumptions)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if not stress_res.policy_decision.accepted:
            st.info(
                "This event was not executed as a formal stress test. "
                "Select a scenario meeting both policy thresholds to view "
                "portfolio valuation and loss attribution."
            )
            st.stop()

        st.markdown("---")
        # Summary KPI Row
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-title">Baseline Balance Sheet Value</div>
                <div class="metric-number">${stress_res.total_baseline_portfolio_value/1e6:.1f}M</div>
                <div class="metric-footnote">Pre-Shock Net Asset Value</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Stressed Balance Sheet Value</div>
                <div class="metric-number">${stress_res.total_stressed_portfolio_value/1e6:.1f}M</div>
                <div class="metric-footnote">Post-Shock Valuated Book</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Capital At Risk (Estimated Loss)</div>
                <div class="metric-number" style="color: #EF4444;">-${stress_res.total_pnl_loss/1e6:.2f}M</div>
                <div class="metric-footnote">Net Drawdown: {stress_res.total_percentage_loss:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Primary Loss Contributor</div>
                <div class="metric-number" style="font-size: 18px; color: #F97316;">{max(stress_res.asset_class_losses.items(), key=lambda x: x[1])[0]}</div>
                <div class="metric-footnote">Largest Asset-Class Vulnerability</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        c_flow1, c_flow2 = st.columns(2)
        with c_flow1:
            fig_waterfall = go.Figure(go.Waterfall(
                name="Capital Flow",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                x=["Baseline NAV", "Equity Loss", "Bond Loss", "Loan Provisions", "Swap Loss", "Stressed NAV"],
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
                connector={"line": {"color": "#9FB3C8"}},
                decreasing={"marker": {"color": "#EF4444"}},
                totals={"marker": {"color": "#2563EB"}},
                increasing={"marker": {"color": "#10B981"}}
            ))
            fig_waterfall.update_layout(
                title="Wholesale Portfolio Capital Impact Waterfall ($)",
                template="plotly_white",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(
                fig_waterfall,
                width="stretch",
                key="stress-waterfall-chart",
            )

        with c_flow2:
            pie_data = [{"Asset Class": k, "Loss": v} for k, v in stress_res.asset_class_losses.items() if v > 0]
            pie_df = pd.DataFrame(pie_data, columns=["Asset Class", "Loss"])
            fig_pie = px.pie(
                pie_df,
                names="Asset Class",
                values="Loss",
                title="Risk Loss Contribution by Asset Class",
                color_discrete_sequence=["#EF4444", "#F97316", "#EAB308", "#38BDF8"],
                hole=0.45,
                template="plotly_white"
            )
            fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, width="stretch", key="stress-loss-pie-chart")

        st.markdown("##### Position-Level Impact Detail")
        position_columns = [
            "position_id", "asset_type", "ticker", "issuer", "baseline_value",
            "stressed_value", "pnl_loss", "percentage_change", "applied_shock",
        ]
        df_positions = pd.DataFrame(
            [p.model_dump() for p in stress_res.position_impacts],
            columns=position_columns,
        )
        disp_schedule = df_positions[["position_id", "asset_type", "ticker", "issuer", "baseline_value", "stressed_value", "pnl_loss", "percentage_change", "applied_shock"]].copy()
        disp_schedule["baseline_value"] = disp_schedule["baseline_value"].map("${:,.0f}".format)
        disp_schedule["stressed_value"] = disp_schedule["stressed_value"].map("${:,.0f}".format)
        disp_schedule["pnl_loss"] = disp_schedule["pnl_loss"].map(lambda x: f"-${x:,.0f}" if x > 0 else "$0")
        disp_schedule["percentage_change"] = disp_schedule["percentage_change"].map("{:.2f}%".format)

        st.dataframe(
            disp_schedule.rename(columns={
                "position_id": "Position ID", "asset_type": "Asset Class", "ticker": "Symbol",
                "issuer": "Counterparty / Issuer", "baseline_value": "Baseline Value", "stressed_value": "Stressed Value",
                "pnl_loss": "Drawdown P&L", "percentage_change": "% Shift", "applied_shock": "Applied Stress Sensitivity"
            }),
            width="stretch",
            hide_index=True
        )

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("##### Loss by Issuer")
            issuer_df = pd.DataFrame(
                [
                    {"Issuer": issuer, "Loss": loss}
                    for issuer, loss in sorted(
                        stress_res.issuer_losses.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
                columns=["Issuer", "Loss"],
            )
            issuer_df["Loss"] = issuer_df["Loss"].map("${:,.0f}".format)
            st.dataframe(issuer_df, width="stretch", hide_index=True)
        with a2:
            st.markdown("##### Loss by Sector")
            sector_df = pd.DataFrame(
                [
                    {"Sector": sector, "Loss": loss}
                    for sector, loss in sorted(
                        stress_res.sector_losses.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ],
                columns=["Sector", "Loss"],
            )
            sector_df["Loss"] = sector_df["Loss"].map("${:,.0f}".format)
            st.dataframe(sector_df, width="stretch", hide_index=True)

# -------------------------------------------------------------
# WORKSPACE 4: Admin Only - Securities Master & Coverage Directory
# -------------------------------------------------------------
elif selected_workspace == "4. Securities Master & Coverage Directory" and is_admin:
    st.markdown("### Institutional Securities Master & Coverage Directory")
    st.caption("Administrative reference database containing ticker aliases, sector classifications, credit ratings, and market beta sensitivities utilized by the entity resolution engine.")

    df_master = pd.DataFrame(security_master)
    df_master["aliases"] = df_master["aliases"].apply(lambda al: ", ".join(al[:5]) + ("..." if len(al) > 5 else ""))
    
    st.dataframe(
        df_master.rename(columns={
            "ticker": "Symbol", "name": "Company", "sector": "Industry Sector",
            "aliases": "Also Known As", "beta": "Market Beta", "credit_rating": "Credit Rating"
        }),
        width="stretch",
        hide_index=True
    )

# -------------------------------------------------------------
# FOOTER: Governance & Compliance Note (SR 11-7 / BCBS 239)
# -------------------------------------------------------------
st.markdown("""<div class="gov-panel"><div class="gov-grid">
<div>
<div class="gov-title">Model Governance &amp; Regulatory Compliance</div>
<div class="gov-sub">How this platform supports supervisory expectations for model risk and risk data.</div>
</div>
<div>
<span class="gov-badge">FEDERAL RESERVE SR 11-7</span>
<div class="gov-item-title">Model Risk Management &mdash; Screening Policy</div>
<div class="gov-item-text">Unconfirmed or minor items (for example rumors about accessories, or a severity score below 7.0) are recorded and excluded, with the reason kept for audit, so they cannot trigger unwarranted balance-sheet changes.</div>
</div>
<div>
<span class="gov-badge">BASEL BCBS 239</span>
<div class="gov-item-title">Risk Data Aggregation &mdash; Full Traceability</div>
<div class="gov-item-text">Every figure can be traced from the original news text and its sentiment scores down to the capital loss on each individual position.</div>
</div>
</div></div>""", unsafe_allow_html=True)