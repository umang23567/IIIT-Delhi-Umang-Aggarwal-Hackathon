import json
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ResolvedEntity(BaseModel):
    ticker: str
    company_name: str
    sector: str
    beta: float
    credit_rating: str
    confidence: float
    matched_alias: str
    mention_span: str

class EntityResolver:
    """
    Resolves mentions of organizations, tickers, and key leadership
    to normalized security master tickers with confidence scoring.
    """
    def __init__(self, master_path: str = "data/reference/security_master.json", enable_ner: bool = False):
        with open(master_path, "r", encoding="utf-8") as f:
            self.master_data: List[Dict[str, Any]] = json.load(f)
        
        # Build lookup indices
        self.alias_to_entity: Dict[str, Dict[str, Any]] = {}
        for item in self.master_data:
            ticker = item["ticker"]
            self.alias_to_entity[ticker.lower()] = item
            for alias in item["aliases"]:
                self.alias_to_entity[alias.lower()] = item
        self.ner_pipeline = None
        if enable_ner:
            try:
                from transformers import pipeline
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                )
            except Exception as exc:
                logger.warning("NER model unavailable; using security-master matching: %s", exc)

    def resolve(self, text: str) -> List[ResolvedEntity]:
        resolved: List[ResolvedEntity] = []
        seen_tickers = set()
        lowered = text.lower()
        ner_mentions = []
        if self.ner_pipeline:
            try:
                ner_mentions = [
                    item["word"] for item in self.ner_pipeline(text[:1500])
                    if item.get("entity_group") == "ORG"
                ]
            except Exception as exc:
                logger.warning("NER inference failed; using security-master matching: %s", exc)
        candidate_text = f"{text} {' '.join(ner_mentions)}"

        # Check ticker with word boundaries (e.g. \bAAPL\b or \bBA\b)
        for item in self.master_data:
            ticker = item["ticker"]
            pattern = rf"\b{ticker}\b"
            if re.search(pattern, candidate_text): # exact case match for tickers
                if ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    resolved.append(ResolvedEntity(
                        ticker=ticker,
                        company_name=item["name"],
                        sector=item["sector"],
                        beta=item["beta"],
                        credit_rating=item["credit_rating"],
                        confidence=0.98,
                        matched_alias=ticker,
                        mention_span=ticker
                    ))

        # Check company names and aliases
        for alias, item in self.alias_to_entity.items():
            ticker = item["ticker"]
            if ticker in seen_tickers:
                continue

            # Word boundary regex for alias
            pattern = rf"\b{re.escape(alias)}\b"
            match = re.search(pattern, candidate_text.lower())
            if match:
                confidence = 0.95 if alias.lower() == item["name"].lower() else 0.88
                seen_tickers.add(ticker)
                span = text[match.start():match.end()]
                resolved.append(ResolvedEntity(
                    ticker=ticker,
                    company_name=item["name"],
                    sector=item["sector"],
                    beta=item["beta"],
                    credit_rating=item["credit_rating"],
                    confidence=confidence,
                    matched_alias=alias,
                    mention_span=span
                ))

        return resolved
