import json
from pathlib import Path

from src.models.event_classifier import EventTaxonomyClassifier
from src.models.impact_engine import ImpactScoringEngine
from src.models.sentiment_engine import FinBERTSentimentEngine
from src.pipeline.entity_resolver import EntityResolver


FIXTURE = (
    Path(__file__).parents[1]
    / "data"
    / "evaluation"
    / "evaluation_samples.json"
)


def test_compact_labeled_evaluation_fixture():
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sentiment = FinBERTSentimentEngine(enable_transformer=False, allow_fallback=True)
    events = EventTaxonomyClassifier(enable_zero_shot=False)
    resolver = EntityResolver()
    impact = ImpactScoringEngine()

    sentiment_correct = 0
    event_correct = 0
    entity_correct = 0
    high_impact_correct = 0

    for sample in samples:
        sentiment_result = sentiment.analyze(sample["text"])
        event_result = events.classify(sample["text"])
        entities = resolver.resolve(sample["text"])
        entity_ticker = entities[0].ticker if entities else None
        score = impact.compute(
            event_type=event_result.event_type,
            event_confidence=event_result.confidence,
            sentiment_score=sentiment_result.sentiment_score,
            sentiment_conf=sentiment_result.confidence,
            has_direct_entity=bool(entities),
            source_name="Reuters Financial Wire",
        )

        sentiment_correct += sentiment_result.sentiment_label == sample["sentiment"]
        event_correct += event_result.event_type == sample["event_type"]
        entity_correct += entity_ticker == sample["ticker"]
        high_impact_correct += (score.impact_score >= 7.0) == (sample["impact_score"] >= 7)

    count = len(samples)
    assert sentiment_correct / count >= 0.80
    assert event_correct / count >= 0.80
    assert entity_correct / count >= 0.80
    assert high_impact_correct / count >= 0.80
