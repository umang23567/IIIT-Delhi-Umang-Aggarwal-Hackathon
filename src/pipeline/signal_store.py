import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class StructuredRiskSignal(BaseModel):
    signal_id: str
    article_id: str
    published_at: str
    source_name: str
    source_type: str
    title: str
    url: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    sentiment_score: float # -1.0 to 1.0
    sentiment_label: str   # POSITIVE, NEUTRAL, NEGATIVE
    sentiment_probs: Dict[str, float]
    event_type: str
    event_confidence: float
    evidence_text: str
    impact_score: float    # 1.0 to 10.0
    impact_confidence: float
    impact_drivers: List[str]
    materiality: str
    model_version: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

class RiskSignalStore:
    """
    Central database store for immutable raw articles and structured risk signals.
    """
    def __init__(self, db_path: str = "data/risk_engine.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Raw articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_articles (
                article_id TEXT PRIMARY KEY,
                source_type TEXT,
                source_name TEXT,
                url TEXT,
                title TEXT,
                body TEXT,
                published_at TEXT,
                fetched_at TEXT,
                content_hash TEXT
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_signals (
                signal_id TEXT PRIMARY KEY,
                article_id TEXT,
                published_at TEXT,
                source_name TEXT,
                source_type TEXT,
                title TEXT,
                url TEXT,
                ticker TEXT,
                company_name TEXT,
                sector TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                sentiment_probs TEXT,
                event_type TEXT,
                event_confidence REAL,
                evidence_text TEXT,
                impact_score REAL,
                impact_confidence REAL,
                impact_drivers TEXT,
                materiality TEXT,
                model_version TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_signal(self, signal: StructuredRiskSignal):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO risk_signals (
                signal_id, article_id, published_at, source_name, source_type,
                title, url, ticker, company_name, sector, sentiment_score,
                sentiment_label, sentiment_probs, event_type, event_confidence,
                evidence_text, impact_score, impact_confidence, impact_drivers,
                materiality, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.signal_id, signal.article_id, signal.published_at, signal.source_name,
            signal.source_type, signal.title, signal.url, signal.ticker, signal.company_name,
            signal.sector, signal.sentiment_score, signal.sentiment_label,
            json.dumps(signal.sentiment_probs), signal.event_type, signal.event_confidence,
            signal.evidence_text, signal.impact_score, signal.impact_confidence,
            json.dumps(signal.impact_drivers), signal.materiality, signal.model_version,
            signal.created_at
        ))
        conn.commit()
        conn.close()

    def get_all_signals(self, limit: int = 100) -> List[StructuredRiskSignal]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT signal_id, article_id, published_at, source_name, source_type,
                   title, url, ticker, company_name, sector, sentiment_score,
                   sentiment_label, sentiment_probs, event_type, event_confidence,
                   evidence_text, impact_score, impact_confidence, impact_drivers,
                   materiality, model_version, created_at
            FROM risk_signals
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        signals = []
        for r in rows:
            signals.append(StructuredRiskSignal(
                signal_id=r[0], article_id=r[1], published_at=r[2], source_name=r[3],
                source_type=r[4], title=r[5], url=r[6], ticker=r[7], company_name=r[8],
                sector=r[9], sentiment_score=r[10], sentiment_label=r[11],
                sentiment_probs=json.loads(r[12]), event_type=r[13], event_confidence=r[14],
                evidence_text=r[15], impact_score=r[16], impact_confidence=r[17],
                impact_drivers=json.loads(r[18]), materiality=r[19], model_version=r[20],
                created_at=r[21]
            ))
        return signals
