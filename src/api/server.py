from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from src.models.nlp_engine import NLPFinancialRiskEngine
from src.pipeline.cleaner import RawArticle
from src.pipeline.signal_store import RiskSignalStore, StructuredRiskSignal
from src.pipeline.feed_adapters import ReplayAdapter, RSSFeedAdapter
from src.rebalancer.rebalancer_engine import TacticalIndexRebalancer, RebalanceRunResult
from src.stress.stress_engine import WholesalePortfolioStressEngine, StressTestResult

app = FastAPI(
    title="S&P Global & CRISIL - AI/NLP Financial Risk Engine & Stress Testing API",
    description="Unified API exposing financial NLP risk signals, dynamic tactical index rebalancing, and strategic wholesale portfolio stress testing.",
    version="1.0.0"
)

# Initialize engines
store = RiskSignalStore()
nlp_engine = NLPFinancialRiskEngine(enable_transformer=False) # FastAPI default lightweight; UI loads transformer
rebalancer = TacticalIndexRebalancer()
stress_engine = WholesalePortfolioStressEngine()

class IngestArticleRequest(BaseModel):
    article_id: str
    source_type: str
    source_name: str
    url: str
    title: str
    body: str
    published_at: str

class StressTestRequest(BaseModel):
    signal_title: str
    event_type: str
    impact_score: float
    confidence: float
    source_name: str
    affected_ticker: Optional[str] = None

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI/NLP Financial Risk Engine",
        "endpoints": [
            "/api/signals",
            "/api/rebalance",
            "/api/stress-test",
            "/internal/ingest",
            "/internal/replay-batch"
        ]
    }

@app.get("/api/signals", response_model=List[StructuredRiskSignal])
def get_signals(
    limit: int = Query(50, ge=1, le=200),
    ticker: Optional[str] = None,
    min_impact: Optional[float] = None
):
    signals = store.get_all_signals(limit=limit)
    if ticker:
        signals = [s for s in signals if s.ticker == ticker]
    if min_impact is not None:
        signals = [s for s in signals if s.impact_score >= min_impact]
    return signals

@app.post("/internal/ingest")
def ingest_article(req: IngestArticleRequest):
    raw = RawArticle(**req.dict())
    signal = nlp_engine.process_article(raw)
    if not signal:
        return {"status": "deduplicated_or_skipped", "article_id": req.article_id}
    return {"status": "processed", "signal": signal}

@app.post("/internal/replay-batch")
def replay_batch():
    adapter = ReplayAdapter()
    articles = adapter.fetch_articles()
    signals = nlp_engine.process_batch(articles)
    return {"status": "success", "processed_signals": len(signals)}

@app.get("/api/rebalance", response_model=RebalanceRunResult)
def run_rebalancing(alpha: float = Query(0.5, ge=0.0, le=2.0)):
    signals = store.get_all_signals(limit=100)
    return rebalancer.rebalance(signals, alpha=alpha)

@app.post("/api/stress-test", response_model=StressTestResult)
def run_portfolio_stress(req: StressTestRequest):
    result = stress_engine.run_stress_test(
        signal_title=req.signal_title,
        event_type=req.event_type,
        impact_score=req.impact_score,
        confidence=req.confidence,
        source_name=req.source_name,
        affected_ticker=req.affected_ticker
    )
    return result

@app.get("/api/portfolio")
def get_portfolio():
    return stress_engine.portfolio_data
