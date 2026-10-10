from pathlib import Path

from src.pipeline.cleaner import ArticleCleaner, RawArticle
from src.pipeline.signal_store import RiskSignalStore


def test_article_cleaner_removes_html_and_detects_near_duplicate():
    cleaner = ArticleCleaner()
    first = cleaner.process(RawArticle(
        article_id="ART_01",
        source_type="News",
        source_name="Reuters",
        url="https://example.com/1",
        title="<b>Apple</b> Reports Record Earnings",
        body="Apple Inc announced quarterly profits beat expectations.",
        published_at="2026-10-09T10:00:00Z",
    ))
    duplicate = cleaner.process(RawArticle(
        article_id="ART_02",
        source_type="News",
        source_name="Reuters Copy",
        url="https://example.com/2",
        title="Apple Reports Record Earnings",
        body="Apple Inc announced quarterly profits beat expectations.",
        published_at="2026-10-09T10:05:00Z",
    ))

    assert "<b>" not in first.cleaned_title
    assert first.is_duplicate is False
    assert duplicate.is_duplicate is True
    assert duplicate.duplicate_group_id == "ART_01"


def test_raw_article_persistence_deduplicates_url_and_hash(tmp_path: Path):
    store = RiskSignalStore(str(tmp_path / "signals.db"))
    raw = RawArticle(
        article_id="ART_PERSISTED",
        source_type="News",
        source_name="Reuters",
        url="https://example.com/persisted",
        title="Persistent Article",
        body="The article is stored before analysis.",
        published_at="2026-10-09T10:00:00Z",
    )
    raw.content_hash = raw.compute_hash()

    assert store.save_raw_article(raw) is True
    assert store.save_raw_article(raw) is False
