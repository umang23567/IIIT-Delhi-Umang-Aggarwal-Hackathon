"""Signal API. Run: uvicorn src.api:app --reload  (docs at /docs)"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from . import config as C
from .nlp import analyze
from .modules.stress import run_stress

app = FastAPI(title="ContagionLens Risk Engine")


class Text(BaseModel):
    text: str
    source: str = "news"
    followers: int | None = None


class StressReq(BaseModel):
    event_type: str
    impact: float
    entity: str = "MARKET"
    scope: str = "market"
    sentiment: float = -1.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/signals")
def signals(ticker: str | None = None, min_impact: float = 0.0, limit: int = 100):
    df = pd.read_csv(C.OUT / "signals.csv")
    df = df[df.impact >= min_impact]
    if ticker:
        df = df[df.entity == ticker.upper()]
    cols = ["timestamp", "source", "entity", "scope", "sentiment", "event_type", "impact", "credibility", "uncorroborated", "text"]
    return df[cols].tail(limit).to_dict("records")


@app.post("/analyze")
def analyze_text(body: Text):
    a = analyze(body.text, body.source, body.followers)
    a["entities"] = [{"entity": e, "scope": s} for e, s in a["entities"]]
    return a


@app.post("/stress")
def stress(body: StressReq):
    pf = pd.read_csv(C.DATA / "portfolio.csv")
    return run_stress(pf, body.event_type, body.impact, body.entity, body.scope, body.sentiment)
