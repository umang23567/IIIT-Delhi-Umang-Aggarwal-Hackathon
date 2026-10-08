"""Module B: event-driven stress testing of a synthetic wholesale-banking book."""
import numpy as np
import pandas as pd

from .. import config as C
from ..graph import build_adjacency


def detect_triggers(sig: pd.DataFrame, thr: float = C.STRESS_IMPACT_THRESHOLD,
                    min_cred: float = C.STRESS_MIN_CREDIBILITY):
    """Adverse, high-impact events. Low-credibility (e.g. uncorroborated rumours) are blocked, not dropped."""
    adv = sig[(sig.impact >= thr) & (sig.sentiment < 0)]
    ok = adv[adv.credibility >= min_cred]
    blocked = adv[adv.credibility < min_cred].copy()
    blocked["reason"] = np.where(blocked.uncorroborated, "uncorroborated social media", "low source credibility")

    def collapse(df):
        if df.empty:
            return df
        return (df.sort_values("impact", ascending=False)
                  .groupby(["date", "event_type", "entity"], as_index=False).first()
                  .sort_values("impact", ascending=False))
    return collapse(ok), collapse(blocked)


def run_stress(portfolio: pd.DataFrame, event_type: str, impact: float, entity: str = "MARKET",
               scope: str = "market", sentiment: float = -1.0) -> dict:
    tickers = C.TICKERS
    A = build_adjacency(tickers)
    sc = C.SCENARIOS.get(event_type, C.SCENARIOS["Other"])
    s = float(np.clip(impact / 7.0, 0.5, 1.6))
    eq, dy, ds = sc["equity"] * s, sc["dy_bp"] * s, sc["spread_bp"] * s
    pdm = 1 + (sc["pd_mult"] - 1) * s
    idio = np.zeros(len(tickers))
    if sentiment < 0:
        if scope == "company" and entity in tickers:
            v = np.zeros(len(tickers)); v[tickers.index(entity)] = -impact / 10 * 0.25
            idio = v + 0.6 * (A @ v)          # contagion: connected names take second-order hits
        elif scope == "sector":
            idio = np.array([-impact / 10 * 0.12 if C.UNIVERSE[t][1] == entity else 0.0 for t in tickers])
    idio = dict(zip(tickers, idio))

    def eq_shock(r):
        base = sc["sector_equity"].get(r.sector, eq) * s if r.sector in sc["sector_equity"] else eq
        return base + idio[r.entity]

    pf = portfolio.copy()
    pf["eq_shock"] = pf.apply(eq_shock, axis=1)
    bad = pf.entity.map(lambda t: max(0.0, -idio[t]))
    pf["pnl"] = 0.0
    m = pf.asset_type
    pf.loc[m == "Equity", "pnl"] = pf.market_value * pf.eq_shock
    pf.loc[m == "Option", "pnl"] = pf.delta * pf.notional * pf.eq_shock
    pf.loc[m == "Bond", "pnl"] = pf.market_value * (-pf.duration * dy / 1e4
                                                     - pf.spread_duration * (ds + bad * 600) / 1e4)
    pd_new = np.minimum(1.0, pf.pd * pdm * (1 + 4 * bad))
    pf.loc[m == "Loan", "pnl"] = -pf.notional * pf.lgd * (pd_new - pf.pd)
    pf.loc[m == "Swap", "pnl"] = pf.dv01 * dy
    before = float(pf.market_value.sum())
    pnl = float(pf.pnl.sum())
    grp = lambda col: pf.groupby(col).pnl.sum().sort_values().round(0).to_dict()
    return dict(event_type=event_type, impact=impact, entity=entity, scope=scope, scale=round(s, 2),
                value_before=round(before, 0), value_after=round(before + pnl, 0), pnl=round(pnl, 0),
                pnl_pct=round(pnl / before * 100, 3), by_asset_type=grp("asset_type"), by_sector=grp("sector"),
                top_losers=pf.nsmallest(8, "pnl")[["position_id", "asset_type", "entity", "pnl"]]
                .round(0).to_dict("records"))
