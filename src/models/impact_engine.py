from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ImpactEvaluation(BaseModel):
    impact_score: float # 1.0 to 10.0 scale
    confidence: float
    drivers: List[str]
    materiality_level: str # LOW, MEDIUM, HIGH, CRITICAL

class ImpactScoringEngine:
    """
    Computes an explainable 1-10 severity score incorporating:
    - Event Class Base Materiality
    - Sentiment Direction and Strength (FinBERT)
    - Entity Directness (single company vs sector vs macro)
    - Source Reliability Weight
    """
    EVENT_BASE_MATERIALITY = {
        "Credit Event": 7.5,
        "Geopolitical": 7.0,
        "Macroeconomic": 6.8,
        "Regulatory/Legal": 6.2,
        "Supply-Chain Disruption": 5.8,
        "Cybersecurity": 6.0,
        "M&A": 5.0,
        "Product Launch": 2.5,
        "General Financial News": 2.0
    }

    SOURCE_RELIABILITY_WEIGHTS = {
        "reuters financial wire": 1.0,
        "bloomberg markets": 1.0,
        "wall street journal": 0.95,
        "financial times": 0.95,
        "cnbc": 0.85,
        "rss financial news": 0.80,
        "social media / blog": 0.50,
        "fintech rumor board": 0.40
    }

    def compute(
        self,
        event_type: str,
        event_confidence: float,
        sentiment_score: float, # -1.0 to +1.0
        sentiment_conf: float,
        has_direct_entity: bool,
        source_name: str
    ) -> ImpactEvaluation:
        drivers = []
        base = self.EVENT_BASE_MATERIALITY.get(event_type, 3.0)
        drivers.append(f"Base Event Materiality ({event_type}): {base:.1f}/10")

        # Sentiment penalty/multiplier: highly negative sentiment amplifies risk severity
        sentiment_mod = 0.0
        if sentiment_score < -0.5:
            sentiment_mod = 2.0 * abs(sentiment_score)
            drivers.append(f"Severe Negative Sentiment ({sentiment_score:+.2f}): +{sentiment_mod:.1f}")
        elif sentiment_score < -0.1:
            sentiment_mod = 1.0 * abs(sentiment_score)
            drivers.append(f"Moderate Negative Sentiment ({sentiment_score:+.2f}): +{sentiment_mod:.1f}")
        elif sentiment_score > 0.5:
            # Positive news has lower downside risk impact
            sentiment_mod = -1.0
            drivers.append(f"Positive Sentiment ({sentiment_score:+.2f}): -1.0 (Downside dampened)")

        # Entity directness
        entity_mod = 0.0
        if has_direct_entity:
            entity_mod = 0.8
            drivers.append("Direct Named Security Master Exposure: +0.8")
        else:
            drivers.append("Broad Market/Systemic Exposure")

        # Source reliability adjustment
        source_key = source_name.lower().strip()
        rel_weight = 0.70
        for k, v in self.SOURCE_RELIABILITY_WEIGHTS.items():
            if k in source_key:
                rel_weight = v
                break
        drivers.append(f"Source Credibility Factor ({source_name}): {rel_weight:.2f}")

        # Raw calculation
        raw_score = (base + sentiment_mod + entity_mod) * rel_weight

        # Clamp between 1.0 and 10.0
        final_score = round(max(1.0, min(10.0, raw_score)), 2)

        overall_conf = round((event_confidence * 0.5 + sentiment_conf * 0.5) * rel_weight, 3)

        if final_score >= 8.0:
            mat = "CRITICAL"
        elif final_score >= 6.5:
            mat = "HIGH"
        elif final_score >= 4.0:
            mat = "MEDIUM"
        else:
            mat = "LOW"

        return ImpactEvaluation(
            impact_score=final_score,
            confidence=overall_conf,
            drivers=drivers,
            materiality_level=mat
        )
