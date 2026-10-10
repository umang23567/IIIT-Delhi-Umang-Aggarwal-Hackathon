from src.models.event_classifier import EventTaxonomyClassifier
from src.models.impact_engine import ImpactScoringEngine


def test_event_classifier_identifies_credit_event():
    result = EventTaxonomyClassifier(enable_zero_shot=False).classify(
        "Fitch downgrades senior unsecured debt rating to junk amid liquidity strain."
    )

    assert result.event_type == "Credit Event"
    assert result.confidence >= 0.70


def test_impact_engine_scores_material_negative_event():
    result = ImpactScoringEngine().compute(
        event_type="Credit Event",
        event_confidence=0.90,
        sentiment_score=-0.80,
        sentiment_conf=0.85,
        has_direct_entity=True,
        source_name="Reuters Financial Wire",
    )

    assert result.impact_score >= 7.5
    assert result.materiality_level in ["HIGH", "CRITICAL"]
