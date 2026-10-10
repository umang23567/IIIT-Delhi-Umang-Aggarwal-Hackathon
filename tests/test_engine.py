import pytest
from src.pipeline.cleaner import ArticleCleaner, RawArticle
from src.pipeline.entity_resolver import EntityResolver
from src.models.event_classifier import EventTaxonomyClassifier
from src.models.impact_engine import ImpactScoringEngine
from src.stress.stress_engine import WholesalePortfolioStressEngine

def test_article_cleaner_and_dedup():
    cleaner = ArticleCleaner()
    raw1 = RawArticle(
        article_id="ART_01",
        source_type="News",
        source_name="Reuters",
        url="https://example.com/1",
        title="<b>Apple</b> Reports Record Earnings",
        body="Apple Inc announced quarterly profits beat expectations.",
        published_at="2026-10-09T10:00:00Z"
    )
    clean1 = cleaner.process(raw1)
    assert "<b>" not in clean1.cleaned_title
    assert clean1.is_duplicate is False

    # Submit duplicate
    raw2 = RawArticle(
        article_id="ART_02",
        source_type="News",
        source_name="Reuters Copy",
        url="https://example.com/2",
        title="Apple Reports Record Earnings",
        body="Apple Inc announced quarterly profits beat expectations.",
        published_at="2026-10-09T10:05:00Z"
    )
    clean2 = cleaner.process(raw2)
    assert clean2.is_duplicate is True
    assert clean2.duplicate_group_id == "ART_01"

def test_entity_resolver():
    resolver = EntityResolver()
    entities = resolver.resolve("The Boeing Company senior debt was placed on credit watch.")
    assert len(entities) > 0
    assert entities[0].ticker == "BA"

def test_event_classifier():
    classifier = EventTaxonomyClassifier()
    res = classifier.classify("Fitch downgrades senior unsecured debt rating to junk amid liquidity strain.")
    assert res.event_type == "Credit Event"
    assert res.confidence >= 0.70

def test_impact_engine():
    engine = ImpactScoringEngine()
    impact = engine.compute(
        event_type="Credit Event",
        event_confidence=0.90,
        sentiment_score=-0.80,
        sentiment_conf=0.85,
        has_direct_entity=True,
        source_name="Reuters Financial Wire"
    )
    assert impact.impact_score >= 7.5
    assert impact.materiality_level in ["HIGH", "CRITICAL"]

def test_stress_testing_math():
    engine = WholesalePortfolioStressEngine()
    res = engine.run_stress_test(
        signal_title="Boeing Debt Downgraded",
        event_type="Credit Event",
        impact_score=8.5,
        confidence=0.88,
        source_name="Reuters",
        affected_ticker="BA"
    )
    assert res.policy_decision.accepted is True
    assert res.total_pnl_loss > 0.0
    assert res.total_stressed_portfolio_value < res.total_baseline_portfolio_value
