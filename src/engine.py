"""Risk Engine: ingest (news + tweets) -> NLP -> structured risk signals (CSV/JSON/API)."""
import json
import re

import numpy as np
import pandas as pd

from .config import DATA, OUT, TICKERS, UNIVERSE, SIGNAL_HALF_LIFE_DAYS
from .nlp import analyze, analyze_naive

SCOPE_W = {"company": 1.0, "sector": 0.6, "market": 0.3}


def load_documents() -> pd.DataFrame:
    """Unified schema from any number of sources. Needs columns: text, timestamp (+followers)."""
    parts = []
    for name, src in [("news.csv", "news"), ("tweets.csv", "twitter")]:
        p = DATA / name
        if p.exists():
            d = pd.read_csv(p, parse_dates=["timestamp"])
            d["source"] = src
            parts.append(d)
    if not parts:
        raise FileNotFoundError("No data found. Run: python main.py gen-data")
    docs = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return dedupe(docs)


def _norm(t: str) -> set:
    t = re.sub(r"^rt\s+", "", t.lower())
    return set(re.findall(r"[a-z]{3,}", t))


def dedupe(docs: pd.DataFrame, thr: float = 0.8) -> pd.DataFrame:
    """Drop near-duplicate posts (retweets/reposts) within the same day via Jaccard similarity."""
    keep, seen = [], {}
    for i, r in docs.iterrows():
        toks, day = _norm(r.text), r.timestamp.date()
        bucket = seen.setdefault(day, [])
        if any(len(toks & o) / max(1, len(toks | o)) >= thr for o in bucket):
            continue
        bucket.append(toks)
        keep.append(i)
    return docs.loc[keep].reset_index(drop=True)


def run(mode: str = "engine", save: bool = True):
    """Returns (doc_level_df, signals_df). mode='naive' gives the baseline."""
    fn = analyze if mode == "engine" else analyze_naive
    docs = load_documents()
    rows = []
    for r in docs.itertuples():
        a = fn(r.text, r.source, getattr(r, "followers", None))
        rows.append(dict(doc_id=r.doc_id, timestamp=r.timestamp, source=r.source, text=r.text,
                         true_event=getattr(r, "true_event", None), true_sentiment=getattr(r, "true_sentiment", None),
                         is_rumor=getattr(r, "is_rumor", False), **{k: v for k, v in a.items() if k != "entities"},
                         entities=a["entities"]))
    doc_df = pd.DataFrame(rows)
    sig = doc_df.explode("entities").reset_index(drop=True)
    sig["entity"] = sig.entities.map(lambda e: e[0])
    sig["scope"] = sig.entities.map(lambda e: e[1])
    sig = sig.drop(columns="entities")
    sig["date"] = sig.timestamp.dt.normalize()
    sig["uncorroborated"] = False
    if mode == "engine":
        sig = _corroborate(sig)
    sig["eff_weight"] = sig.credibility * sig.impact / 10
    sig.insert(0, "signal_id", range(1, len(sig) + 1))
    if save:
        OUT.mkdir(exist_ok=True, parents=True)
        suffix = "" if mode == "engine" else "_naive"
        sig.to_csv(OUT / f"signals{suffix}.csv", index=False)
        doc_df.drop(columns="entities").to_csv(OUT / f"docs_scored{suffix}.csv", index=False)
        if mode == "engine":
            cols = ["signal_id", "timestamp", "source", "entity", "scope", "sentiment", "event_type",
                    "impact", "credibility", "uncorroborated", "text"]
            (OUT / "signals.json").write_text(sig[cols].assign(timestamp=sig.timestamp.astype(str))
                                              .to_json(orient="records", indent=1))
    return doc_df, sig


def _corroborate(sig: pd.DataFrame) -> pd.DataFrame:
    """Rumour filter: a strong social-media signal with no same-direction news on the same
    entity within +-24h has its credibility halved and is flagged."""
    news = sig[sig.source == "news"]
    sig = sig.copy()
    for i, r in sig[(sig.source == "twitter") & (sig.sentiment.abs() >= 0.3)].iterrows():
        pool = news[news.entity == r.entity]  # market-wide news does not vouch for a company-specific claim
        pool = pool[(pool.timestamp - r.timestamp).abs() <= pd.Timedelta(hours=24)]
        if not (np.sign(pool.sentiment) == np.sign(r.sentiment)).any():
            sig.at[i, "credibility"] = round(r.credibility * 0.5, 3)
            sig.at[i, "uncorroborated"] = True
    return sig


def daily_scores(sig: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate signals to a (date x ticker) exposure-weighted sentiment score with EWMA decay."""
    raw = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    sector_members = {}
    for t, (_, s, _) in UNIVERSE.items():
        sector_members.setdefault(s, []).append(t)
    for r in sig.itertuples():
        if r.date not in raw.index:
            continue
        if r.scope == "company" and r.entity in raw.columns:
            tg = [r.entity]
        elif r.scope == "sector":
            tg = sector_members.get(r.entity, [])
        else:
            tg = TICKERS
        raw.loc[r.date, tg] += SCOPE_W[r.scope] * r.credibility * r.sentiment * r.impact / 10
    decay = 0.5 ** (1 / SIGNAL_HALF_LIFE_DAYS)
    out, prev = raw.copy(), np.zeros(len(TICKERS))
    for i in range(len(raw)):
        prev = decay * prev + raw.iloc[i].values
        out.iloc[i] = prev
    return out
