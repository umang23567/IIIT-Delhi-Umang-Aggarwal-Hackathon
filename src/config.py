"""Central configuration: universe, taxonomy, contagion graph, stress scenarios."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
DOCS = ROOT / "docs"

# ticker -> (name, sector, aliases used for entity extraction)
UNIVERSE = {
    "AAPL": ("Apple", "Technology", ["apple"]),
    "MSFT": ("Microsoft", "Technology", ["microsoft"]),
    "GOOGL": ("Alphabet", "Technology", ["alphabet", "google"]),
    "NVDA": ("Nvidia", "Technology", ["nvidia"]),
    "AMZN": ("Amazon", "Consumer", ["amazon"]),
    "TSLA": ("Tesla", "Consumer", ["tesla"]),
    "WMT": ("Walmart", "Consumer", ["walmart"]),
    "JPM": ("JPMorgan", "Financials", ["jpmorgan", "jp morgan"]),
    "BAC": ("Bank of America", "Financials", ["bank of america"]),
    "GS": ("Goldman Sachs", "Financials", ["goldman"]),
    "XOM": ("Exxon Mobil", "Energy", ["exxon"]),
    "CVX": ("Chevron", "Energy", ["chevron"]),
    "PFE": ("Pfizer", "Healthcare", ["pfizer"]),
    "JNJ": ("Johnson & Johnson", "Healthcare", ["johnson & johnson", "j&j"]),
    "BA": ("Boeing", "Industrials", ["boeing"]),
}
TICKERS = list(UNIVERSE)
SECTORS = sorted({v[1] for v in UNIVERSE.values()})

SECTOR_KEYWORDS = {
    "Energy": ["oil", "crude", "opec", "energy"],
    "Financials": ["banks", "banking"],
    "Technology": ["chip", "semiconductor", "tech stocks"],
    "Healthcare": ["drug", "fda", "pharma"],
    "Industrials": ["aerospace", "airline", "aviation"],
    "Consumer": ["retail", "consumer spending"],
}

EVENT_TYPES = [
    "Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
    "Product Launch", "Earnings", "Regulatory/Legal", "Cyber/Operational", "Other",
]

# Base severity (1-10) before sentiment / magnitude adjustments
EVENT_BASE_IMPACT = {
    "Geopolitical": 6.5, "Macroeconomic": 6.0, "Credit Event": 7.0,
    "Merger/Acquisition": 5.5, "Product Launch": 3.5, "Earnings": 5.0,
    "Regulatory/Legal": 5.5, "Cyber/Operational": 6.0, "Other": 1.5,
}

# Directed exposure graph: (source, target, weight) = risk of source leaks to target.
# ASSUMPTION: hand-built from public knowledge of sector links (supply chain,
# peer banks, lending); weights are illustrative, not estimated.
MANUAL_EDGES = [
    ("NVDA", "MSFT", 0.35), ("NVDA", "GOOGL", 0.30), ("NVDA", "AMZN", 0.30),
    ("NVDA", "TSLA", 0.20), ("MSFT", "AMZN", 0.15), ("AAPL", "GOOGL", 0.15),
    ("XOM", "CVX", 0.50), ("CVX", "XOM", 0.50),
    ("JPM", "BAC", 0.50), ("BAC", "JPM", 0.50),
    ("JPM", "GS", 0.40), ("GS", "JPM", 0.40),
    ("BAC", "GS", 0.30), ("GS", "BAC", 0.30),
    ("PFE", "JNJ", 0.40), ("JNJ", "PFE", 0.40),
    ("XOM", "BA", 0.15), ("JPM", "XOM", 0.20), ("JPM", "TSLA", 0.15),
    ("AMZN", "WMT", 0.20), ("WMT", "AMZN", 0.15),
]
SECTOR_PEER_WEIGHT = 0.15   # default link between same-sector names
CONTAGION_LAMBDA = 0.5      # per-hop decay
NEG_AMPLIFY, POS_DAMPEN = 1.3, 0.7   # bad news travels further than good news

# Stress scenarios keyed by event type. Shocks scale with impact/7 (clipped 0.5-1.6).
SCENARIOS = {
    "Geopolitical": dict(equity=-0.12, dy_bp=-40, spread_bp=120, pd_mult=1.5,
                         sector_equity={"Energy": 0.08, "Industrials": -0.06}),
    "Macroeconomic": dict(equity=-0.08, dy_bp=150, spread_bp=80, pd_mult=1.3, sector_equity={}),
    "Credit Event": dict(equity=-0.06, dy_bp=-20, spread_bp=200, pd_mult=2.2,
                         sector_equity={"Financials": -0.10}),
    "Regulatory/Legal": dict(equity=-0.03, dy_bp=0, spread_bp=30, pd_mult=1.1, sector_equity={}),
    "Cyber/Operational": dict(equity=-0.04, dy_bp=0, spread_bp=40, pd_mult=1.15, sector_equity={}),
    "Merger/Acquisition": dict(equity=-0.01, dy_bp=0, spread_bp=10, pd_mult=1.0, sector_equity={}),
    "Earnings": dict(equity=-0.02, dy_bp=0, spread_bp=15, pd_mult=1.05, sector_equity={}),
    "Product Launch": dict(equity=0.0, dy_bp=0, spread_bp=0, pd_mult=1.0, sector_equity={}),
    "Other": dict(equity=0.0, dy_bp=0, spread_bp=0, pd_mult=1.0, sector_equity={}),
}

STRESS_IMPACT_THRESHOLD = 7.0
STRESS_MIN_CREDIBILITY = 0.6
SIGNAL_HALF_LIFE_DAYS = 2.0
