import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RebalancePosition(BaseModel):
    ticker: str
    company_name: str
    sector: str
    initial_weight: float
    target_weight: float
    weight_delta: float
    current_sentiment: float
    signal_count: int

class RebalanceRunResult(BaseModel):
    timestamp: str
    alpha: float
    total_turnover: float
    positions: List[RebalancePosition]
    summary: str

class TacticalIndexRebalancer:
    """
    Module A: Dynamic tactical index rebalancing based on real-time NLP sentiment.
    Formula:
        w_raw_i = w_0_i * (1 + alpha * sentiment_i)
        w_target_i = w_raw_i / sum(w_raw)
    Clamped to min/max risk bounds per asset (e.g., 2% to 18%).
    """
    def __init__(self, master_path: str = "data/security_master.json", alpha: float = 0.5):
        self.alpha = alpha
        with open(master_path, "r", encoding="utf-8") as f:
            self.master_data = json.load(f)
        
        # Select benchmark universe of 12 stocks
        self.universe = self.master_data[:12]
        self.initial_weight = 1.0 / len(self.universe)
        
        # In-memory history of weights over time
        self.history: List[Dict[str, Any]] = []

    def rebalance(self, signals: List[Any], alpha: Optional[float] = None) -> RebalanceRunResult:
        if alpha is not None:
            self.alpha = alpha

        # Aggregate sentiment by ticker
        ticker_sentiment: Dict[str, List[float]] = {item["ticker"]: [] for item in self.universe}
        for sig in signals:
            t = getattr(sig, "ticker", None) or (sig.get("ticker") if isinstance(sig, dict) else None)
            s = getattr(sig, "sentiment_score", 0.0) if hasattr(sig, "sentiment_score") else sig.get("sentiment_score", 0.0)
            if t and t in ticker_sentiment:
                ticker_sentiment[t].append(s)

        raw_weights = {}
        for item in self.universe:
            t = item["ticker"]
            scores = ticker_sentiment[t]
            avg_s = sum(scores) / len(scores) if scores else 0.0
            
            # Sentiment adjustment
            adjusted = self.initial_weight * (1.0 + self.alpha * avg_s)
            # Bound check: between 1% and 25%
            raw_weights[t] = max(0.01, min(0.25, adjusted))

        # Normalize to sum exactly to 1.0
        total_raw = sum(raw_weights.values())
        normalized_weights = {t: w / total_raw for t, w in raw_weights.items()}

        turnover = 0.0
        positions = []
        for item in self.universe:
            t = item["ticker"]
            target_w = normalized_weights[t]
            delta = target_w - self.initial_weight
            turnover += abs(delta)
            scores = ticker_sentiment[t]
            avg_s = sum(scores) / len(scores) if scores else 0.0

            positions.append(RebalancePosition(
                ticker=t,
                company_name=item["name"],
                sector=item["sector"],
                initial_weight=round(self.initial_weight, 4),
                target_weight=round(target_w, 4),
                weight_delta=round(delta, 4),
                current_sentiment=round(avg_s, 3),
                signal_count=len(scores)
            ))

        positions.sort(key=lambda x: x.target_weight, reverse=True)
        turnover = round(turnover / 2.0, 4) # One-way turnover

        result = RebalanceRunResult(
            timestamp="",
            alpha=self.alpha,
            total_turnover=turnover,
            positions=positions,
            summary=f"Rebalanced {len(positions)} equities with alpha={self.alpha}. Turnover: {turnover*100:.2f}%"
        )
        return result
