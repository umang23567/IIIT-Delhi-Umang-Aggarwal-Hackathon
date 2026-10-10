import logging
from typing import List, Optional
from src.pipeline.cleaner import ArticleCleaner, RawArticle
from src.pipeline.entity_resolver import EntityResolver, ResolvedEntity
from src.models.sentiment_engine import FinBERTSentimentEngine
from src.models.event_classifier import EventTaxonomyClassifier
from src.models.impact_engine import ImpactScoringEngine
from src.pipeline.signal_store import RiskSignalStore, StructuredRiskSignal

logger = logging.getLogger(__name__)

class NLPFinancialRiskEngine:
    """
    Unified AI/NLP Financial Risk Engine.
    Executes:
    1. Cleaning & Deduplication
    2. Entity Resolution (Security Master)
    3. FinBERT Financial Sentiment Analysis
    4. 8-Class Event Classification & Evidence Extraction
    5. Multi-Factor Impact Severity Scoring (1-10)
    6. Persistence into Global Signal Database
    """
    def __init__(
        self,
        enable_transformer: bool = True,
        allow_sentiment_fallback: bool = False,
        enable_ner: bool = False,
    ):
        self.cleaner = ArticleCleaner()
        self.entity_resolver = EntityResolver(enable_ner=enable_ner)
        self.sentiment_engine = FinBERTSentimentEngine(
            enable_transformer=enable_transformer,
            allow_fallback=allow_sentiment_fallback,
        )
        self.event_classifier = EventTaxonomyClassifier()
        self.impact_engine = ImpactScoringEngine()
        self.store = RiskSignalStore()

    def process_article(self, raw: RawArticle) -> Optional[StructuredRiskSignal]:
        # Persist the source record before analysis so every signal has an
        # auditable input and duplicates remain suppressed after restarts.
        raw.content_hash = raw.content_hash or raw.compute_hash()
        if not self.store.save_raw_article(raw):
            logger.info("Skipping persisted duplicate article %s", raw.article_id)
            return None

        # 1. Clean and Deduplicate
        clean = self.cleaner.process(raw)
        if clean.is_duplicate:
            logger.info(f"Skipping duplicate article {raw.article_id} (Parent: {clean.duplicate_group_id})")
            return None

        # 2. Entity Resolution
        entities: List[ResolvedEntity] = self.entity_resolver.resolve(clean.cleaned_text)
        primary_entity = entities[0] if entities else None

        # 3. Sentiment Analysis (FinBERT)
        sentiment = self.sentiment_engine.analyze(clean.cleaned_text)

        # 4. Event Classification
        event = self.event_classifier.classify(clean.cleaned_text)

        # 5. Impact Scoring
        has_direct_entity = primary_entity is not None
        impact = self.impact_engine.compute(
            event_type=event.event_type,
            event_confidence=event.confidence,
            sentiment_score=sentiment.sentiment_score,
            sentiment_conf=sentiment.confidence,
            has_direct_entity=has_direct_entity,
            source_name=raw.source_name
        )

        # 6. Build Structured Signal
        signal = StructuredRiskSignal(
            signal_id=f"SIG_{raw.article_id}",
            article_id=raw.article_id,
            published_at=clean.published_at,
            source_name=clean.source_name,
            source_type=clean.source_type,
            title=clean.cleaned_title,
            url=clean.url,
            ticker=primary_entity.ticker if primary_entity else None,
            company_name=primary_entity.company_name if primary_entity else None,
            sector=primary_entity.sector if primary_entity else None,
            sentiment_score=sentiment.sentiment_score,
            sentiment_label=sentiment.sentiment_label,
            sentiment_probs={
                "positive": sentiment.positive_prob,
                "neutral": sentiment.neutral_prob,
                "negative": sentiment.negative_prob
            },
            event_type=event.event_type,
            event_confidence=event.confidence,
            evidence_text=event.evidence_text,
            impact_score=impact.impact_score,
            impact_confidence=impact.confidence,
            impact_drivers=impact.drivers,
            materiality=impact.materiality_level,
            model_version=sentiment.model_version
        )

        # 7. Persist to DB
        self.store.save_signal(signal)
        self.store.recompute_corroboration()
        return signal

    def process_batch(self, raw_articles: List[RawArticle]) -> List[StructuredRiskSignal]:
        signals = []
        for raw in raw_articles:
            sig = self.process_article(raw)
            if sig:
                signals.append(sig)
        return signals
