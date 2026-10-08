"""NLP core: sentiment, event classification, entity extraction, impact scoring.

Default backend is a transparent finance lexicon + rules (fast, offline, explainable).
An optional FinBERT backend can be enabled with USE_FINBERT=1 (needs transformers+torch).
A deliberately naive baseline is included to quantify the engine's added value.
"""
import math
import os
import re

from .config import (EVENT_BASE_IMPACT, SECTOR_KEYWORDS, TICKERS, UNIVERSE)

POS = {"beats": 2, "beat": 1.5, "surge": 2, "surges": 2, "upbeat": 1.5, "strong": 1.5, "record": 1.5,
       "upgraded": 2, "upgrade": 2, "strengthens": 1.5, "wins": 1.5, "rally": 1.5, "rallies": 1.5,
       "climb": 1.2, "approves": 1.2, "unveils": 1, "cools": 1, "eases": 1.2, "lift": 1.2, "lifts": 1.2,
       "gains": 1, "soar": 2, "boost": 1.5, "huge": 1.2, "bullish": 1.5, "moon": 1.5, "ceasefire": 1.5,
       "fast-tracks": 1.2, "recovery": 1.2, "restores": 1.2, "raises": 0.5, "agrees": 0.5,
       "raises guidance": 2, "drops probe": 2, "rate cut": 1.2, "rate cuts": 1.2, "wins appeal": 2,
       "loving": 1, "producers rally": 1.5}
NEG = {"downgraded": -2, "downgrade": -2, "misses": -2, "miss": -1.5, "default": -2.5, "defaults": -2.5,
       "widen": -1.5, "widens": -1.5, "collapses": -2.5, "collapse": -2.5, "blocked": -1.5, "fine": -1.2,
       "fined": -1.5, "lawsuit": -1.8, "investigation": -1.5, "probe": -1.2, "breach": -2, "cyberattack": -2.5,
       "disrupted": -1.5, "halts": -2, "fire": -1.5, "delays": -1.5, "falls": -1.5, "slide": -1.5,
       "slides": -1.5, "shrink": -1.5, "cuts": -1.5, "concerns": -1, "fears": -1.5, "fear": -1.5,
       "rattle": -1.8, "rattled": -1.8, "escalating": -1.5, "conflict": -1.5, "sanctions": -1.2,
       "tariffs": -1.2, "tensions": -1.2, "recession": -2, "inflation": -0.8, "plunge": -2, "crash": -2.5,
       "slump": -2, "bleeding": -1.5, "rough": -1, "ugh": -0.8, "weigh": -1.2, "mount": -1, "pushback": -1,
       "bankruptcy": -2.5, "weak": -1.2, "hit": -0.8, "curbs": -1.2, "rumors": -0.5, "lenders": -0.3,
       "raises rates": -1.5, "rate hike": -1.5, "falls short": -2, "cuts outlook": -2, "unemployment rises": -1.8,
       "inflation fears": -1.5, "default fears": -2.5}
EMOJI = {"🚀": 1.5, "📈": 1.2, "📉": -1.2, "😬": -1.0, "😱": -1.2, "💀": -1.2, "🚨": -0.8}
NEGATORS = {"not", "no", "never", "without", "isn't", "won't", "avoids", "denies"}

_LEX = {**POS, **NEG}
_LEX_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_LEX, key=len, reverse=True)) + r")(?!\w)")

EVENT_KW = {
    "Geopolitical": ["conflict", "war", "sanction", "ceasefire", "tariff", "trade war", "tension", "invasion",
                     "embargo", "missile", "geopolitic"],
    "Macroeconomic": ["central bank", "rates", "rate cut", "rate hike", "inflation", "recession", "unemployment",
                      "gdp", "jobs report", "yield"],
    "Credit Event": ["downgrad", "upgrad", "default", "bond payment", "credit spread", "debt", "lenders",
                     "bankruptcy", "rating agency", "leverage"],
    "Merger/Acquisition": ["acqui", "merger", "deal", "takeover", "buyout", "talks over"],
    "Product Launch": ["unveil", "launch", "new product", "rollout", "debut"],
    "Earnings": ["earnings", "profit", "revenue", "guidance", "outlook", "margins", "quarterly"],
    "Regulatory/Legal": ["fine", "antitrust", "probe", "lawsuit", "investigation", "regulator", "violation",
                         "appeal", "capital rules", "export curbs", "fda"],
    "Cyber/Operational": ["breach", "cyberattack", "ransomware", "outage", "halts production", "plant fire",
                          "recall", "disrupted"],
}
_EVENT_RE = {k: [re.compile(r"(?<!\w)" + re.escape(w)) for w in v] for k, v in EVENT_KW.items()}

_ALIAS = [(re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)"), t)
          for t, (_, _, al) in UNIVERSE.items() for a in al]
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*(?:billion|bn|b)\b", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|points?)", re.I)
URGENT = ["default", "collapse", "emergency", "bankruptcy", "halts", "cyberattack", "breach", "plunge", "crash"]


def score_sentiment(text: str):
    """Returns (score in [-1,1], n_hits, matched_terms). Handles phrases, negation, emoji."""
    low = text.lower()
    total, terms = 0.0, []
    for m in _LEX_RE.finditer(low):
        w = _LEX[m.group(1)]
        prev = low[:m.start()].split()[-2:]
        if any(p.strip(",.") in NEGATORS for p in prev):
            w = -0.8 * w
        total += w
        terms.append(m.group(1))
    for e, w in EMOJI.items():
        c = text.count(e)
        if c:
            total += w * min(c, 2)
            terms.append(e)
    return math.tanh(total / 3.0), len(terms), terms


def classify_event(text: str):
    """Keyword-weighted event classifier -> (label, confidence, matched keywords)."""
    low = text.lower()
    scores = {k: sum(1 for r in rs if r.search(low)) for k, rs in _EVENT_RE.items()}
    tot = sum(scores.values())
    if tot == 0:
        return "Other", 0.3, []
    best = max(scores, key=lambda k: (scores[k], EVENT_BASE_IMPACT[k]))
    kws = [EVENT_KW[best][i] for i, r in enumerate(_EVENT_RE[best]) if r.search(low)]
    return best, round(scores[best] / tot, 2), kws


def extract_entities(text: str):
    """Returns list of (entity, scope): company > sector > market."""
    low = text.lower()
    found = []
    for t in _CASHTAG.findall(text):
        if t in UNIVERSE and (t, "company") not in found:
            found.append((t, "company"))
    for rx, t in _ALIAS:
        if rx.search(low) and (t, "company") not in found:
            found.append((t, "company"))
    if found:
        return found
    for sec, kws in SECTOR_KEYWORDS.items():
        if any(re.search(r"(?<!\w)" + re.escape(k), low) for k in kws):
            found.append((sec, "sector"))
    return found or [("MARKET", "market")]


def score_impact(text: str, etype: str, sentiment: float) -> float:
    """Severity 1-10: event base x sentiment strength + magnitude cues + urgency."""
    base = EVENT_BASE_IMPACT[etype] * (0.6 + 0.8 * abs(sentiment))
    bonus = 0.0
    for m in _MONEY.findall(text):
        bonus += min(1.5, math.log10(1 + float(m)))
    for m in _PCT.findall(text):
        bonus += min(0.8, float(m) / 2)
    low = text.lower()
    bonus += 0.7 * any(u in low for u in URGENT)
    return round(max(1.0, min(10.0, base + bonus)), 1)


def source_credibility(source: str, followers=None) -> float:
    if source == "news":
        return 0.9
    f = 100 if followers is None or followers != followers else max(float(followers), 10)
    return round(0.3 + 0.3 * min(1.0, math.log10(f) / 6), 3)


_finbert = None


def _finbert_sentiment(text):
    global _finbert
    from transformers import pipeline  # optional dependency
    if _finbert is None:
        _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    out = {d["label"].lower(): d["score"] for d in _finbert(text[:512])[0]}
    return out.get("positive", 0) - out.get("negative", 0)


def analyze(text: str, source: str, followers=None) -> dict:
    sent, hits, terms = score_sentiment(text)
    if os.getenv("USE_FINBERT") == "1":
        try:
            sent = float(_finbert_sentiment(text))
        except Exception:
            pass
    etype, econf, kws = classify_event(text)
    return dict(sentiment=round(sent, 3), event_type=etype, event_conf=econf,
                impact=score_impact(text, etype, sent), credibility=source_credibility(source, followers),
                entities=extract_entities(text), evidence=";".join(terms + kws))


# ---------- naive baseline (for the "vs naive approach" comparison) ----------
_NP = {"beats", "surge", "surges", "upgraded", "strong", "record", "rally", "gains", "huge", "wins"}
_NN = {"downgraded", "default", "misses", "fine", "fined", "lawsuit", "breach", "cyberattack", "falls", "fears",
       "slide", "cuts", "collapses", "halts", "rattle", "recession"}
_NAIVE_EVENT = {"war": "Geopolitical", "inflation": "Macroeconomic", "default": "Credit Event",
                "acquire": "Merger/Acquisition", "launch": "Product Launch", "earnings": "Earnings",
                "lawsuit": "Regulatory/Legal", "breach": "Cyber/Operational"}


def analyze_naive(text: str, source: str, followers=None) -> dict:
    toks = re.findall(r"[a-z\-]+", text.lower())
    s = sum(t in _NP for t in toks) - sum(t in _NN for t in toks)
    s = max(-1.0, min(1.0, s / 2))
    etype = next((v for k, v in _NAIVE_EVENT.items() if k in text.lower()), "Other")
    ents = [(t, "company") for t in _CASHTAG.findall(text) if t in UNIVERSE]
    for rx, t in _ALIAS:
        if rx.search(text.lower()) and (t, "company") not in ents:
            ents.append((t, "company"))
    return dict(sentiment=s, event_type=etype, event_conf=0.5, impact=5.0, credibility=1.0,
                entities=ents or [("MARKET", "market")], evidence="")
