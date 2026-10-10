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
    corroboration_count: int = 1
    corroborating_sources: List[str] = Field(default_factory=list)
    event_cluster_id: str = ""
    novelty_score: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

class RiskSignalStore:
    """
    Central database store for immutable raw articles and structured risk signals.
    """
    def __init__(self, db_path: str = "data/runtime/risk_engine.db"):
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
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_articles_url "
            "ON raw_articles(url) WHERE url IS NOT NULL AND url != ''"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_articles_content_hash "
            "ON raw_articles(content_hash) WHERE content_hash IS NOT NULL AND content_hash != ''"
        )

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
                corroboration_count INTEGER DEFAULT 1,
                corroborating_sources TEXT DEFAULT '[]',
                event_cluster_id TEXT DEFAULT '',
                novelty_score REAL DEFAULT 1.0,
                created_at TEXT
            )
        """)
        for column, definition in (
            ("corroboration_count", "INTEGER DEFAULT 1"),
            ("corroborating_sources", "TEXT DEFAULT '[]'"),
            ("event_cluster_id", "TEXT DEFAULT ''"),
            ("novelty_score", "REAL DEFAULT 1.0"),
        ):
            try:
                cursor.execute(f"ALTER TABLE risk_signals ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()

    def save_raw_article(self, article) -> bool:
        """Persist an input article and report whether it was newly inserted."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO raw_articles (
                    article_id, source_type, source_name, url, title, body,
                    published_at, fetched_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.article_id,
                    article.source_type,
                    article.source_name,
                    article.url,
                    article.title,
                    article.body,
                    article.published_at,
                    article.fetched_at,
                    article.content_hash,
                ),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
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
                materiality, model_version, corroboration_count, corroborating_sources,
                event_cluster_id, novelty_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.signal_id, signal.article_id, signal.published_at, signal.source_name,
            signal.source_type, signal.title, signal.url, signal.ticker, signal.company_name,
            signal.sector, signal.sentiment_score, signal.sentiment_label,
            json.dumps(signal.sentiment_probs), signal.event_type, signal.event_confidence,
            signal.evidence_text, signal.impact_score, signal.impact_confidence,
            json.dumps(signal.impact_drivers), signal.materiality, signal.model_version,
            signal.corroboration_count, json.dumps(signal.corroborating_sources),
            signal.event_cluster_id, signal.novelty_score, signal.created_at
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
                   materiality, model_version, corroboration_count, corroborating_sources,
                   event_cluster_id, novelty_score, created_at
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
                corroboration_count=r[21] or 1,
                corroborating_sources=json.loads(r[22] or "[]"),
                event_cluster_id=r[23] or "",
                novelty_score=r[24] if r[24] is not None else 1.0,
                created_at=r[25]
            ))
        return signals

    def recompute_corroboration(self) -> None:
        """Cluster nearby same-event signals and count independent sources."""
        signals = self.get_all_signals(limit=1000)
        conn = sqlite3.connect(self.db_path)
        try:
            for signal in signals:
                related = [
                    other for other in signals
                    if other.signal_id != signal.signal_id
                    and other.event_type == signal.event_type
                    and other.ticker == signal.ticker
                    and _title_similarity(signal.title, other.title) >= 0.15
                ]
                sources = sorted({signal.source_name, *(item.source_name for item in related)})
                cluster_id = f"{signal.event_type}:{signal.ticker or 'BROAD'}:{_cluster_key(signal.title)}"
                conn.execute(
                    """
                    UPDATE risk_signals
                    SET corroboration_count=?, corroborating_sources=?,
                        event_cluster_id=?, novelty_score=?
                    WHERE signal_id=?
                    """,
                    (len(sources), json.dumps(sources), cluster_id,
                     1.0 if len(sources) == 1 else 1.0 / len(sources), signal.signal_id),
                )
            conn.commit()
        finally:
            conn.close()


def _title_similarity(left: str, right: str) -> float:
    left_words = {word for word in left.lower().split() if len(word) > 3}
    right_words = {word for word in right.lower().split() if len(word) > 3}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(len(left_words), len(right_words))


def _cluster_key(title: str) -> str:
    return "-".join(sorted({word for word in title.lower().split() if len(word) > 4})[:6])
