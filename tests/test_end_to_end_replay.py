import json
from pathlib import Path

from src.models.event_classifier import EventTaxonomyClassifier
from src.models.impact_engine import ImpactScoringEngine
from src.pipeline.cleaner import ArticleCleaner
from src.pipeline.entity_resolver import EntityResolver
from src.pipeline.feed_adapters import ReplayAdapter


def test_replay_feed_produces_expected_entity_and_event_signals():
    expected = {
        item["feed_id"]: item
        for item in json.loads(
            (
                Path(__file__).parents[1]
                / "data"
                / "demo"
                / "replay_feed.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    }
    articles = ReplayAdapter().fetch_articles()
    cleaner = ArticleCleaner()
    resolver = EntityResolver()
    classifier = EventTaxonomyClassifier(enable_zero_shot=False)
    impact_engine = ImpactScoringEngine()

    assert articles
    for article in articles:
        cleaned = cleaner.process(article)
        entities = resolver.resolve(cleaned.cleaned_text)
        event = classifier.classify(cleaned.cleaned_text)
        impact = impact_engine.compute(
            event_type=event.event_type,
            event_confidence=event.confidence,
            sentiment_score=0.0,
            sentiment_conf=0.5,
            has_direct_entity=bool(entities),
            source_name=article.source_name,
        )

        expected_ticker = expected[article.article_id]["expected_ticker"]
        assert (entities[0].ticker if entities else None) == expected_ticker
        assert event.event_type == expected[article.article_id]["expected_event"]
        assert 1.0 <= impact.impact_score <= 10.0
