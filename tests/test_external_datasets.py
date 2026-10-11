import csv
import json
from pathlib import Path

import pytest

from src.models.sentiment_engine import FinBERTSentimentEngine
from src.pipeline.cleaner import ArticleCleaner, RawArticle


EXTERNAL = Path(__file__).parents[1] / "data" / "external"
PHRASEBANK = next(
    (
        path
        for path in (
            EXTERNAL / "financial_phrasebank" / "financial_phrasebank.csv",
            EXTERNAL / "financial_phrasebank.csv",
        )
        if path.exists()
    ),
    EXTERNAL / "financial_phrasebank.csv",
)
_US_NEWS_ROOT = EXTERNAL / "us_financial_news"
US_NEWS_MONTH = next(
    (
        path
        for path in (
            _US_NEWS_ROOT / "2018_03",
            _US_NEWS_ROOT / "2018_03_112b52537b67659ad3609a234388c50a",
            *_US_NEWS_ROOT.glob("2018_03_*"),
            EXTERNAL
            / "US Financial News Articles"
            / "2018_03_112b52537b67659ad3609a234388c50a",
        )
        if path.is_dir()
    ),
    _US_NEWS_ROOT / "2018_03",
)


def _phrasebank_subset(limit_per_label: int = 20) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    with PHRASEBANK.open(encoding="latin-1", newline="") as handle:
        for label, text in csv.reader(handle, delimiter=",", quotechar='"'):
            if label not in {"positive", "negative", "neutral"}:
                continue
            if counts.get(label, 0) >= limit_per_label:
                continue
            rows.append((label.upper(), text))
            counts[label] = counts.get(label, 0) + 1
            if len(counts) == 3 and all(
                count >= limit_per_label for count in counts.values()
            ):
                break
    return rows


def test_financial_phrasebank_sentiment_subset():
    if not PHRASEBANK.exists():
        pytest.skip("Download Financial PhraseBank to data/external first")

    samples = _phrasebank_subset()
    assert len(samples) == 60

    try:
        engine = FinBERTSentimentEngine(
            enable_transformer=True,
            allow_fallback=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"FinBERT is unavailable for external evaluation: {exc}")

    correct = sum(
        engine.analyze(text).sentiment_label == expected
        for expected, text in samples
    )
    accuracy = correct / len(samples)

    # This is a small deterministic smoke sample, not a published benchmark.
    assert accuracy >= 0.60, f"PhraseBank subset accuracy was {accuracy:.1%}"


def _us_news_subset(limit: int = 30) -> list[dict]:
    records = []
    for path in sorted(US_NEWS_MONTH.glob("news_*.json"))[:limit]:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def test_us_financial_news_subset_flows_through_cleaning():
    if not US_NEWS_MONTH.exists():
        pytest.skip("Download US Financial News Articles to data/external first")

    records = _us_news_subset()
    if len(records) < 30:
        pytest.skip("Keep at least 30 extracted news JSON records for this test")
    assert len(records) == 30

    cleaner = ArticleCleaner()
    cleaned = []
    for record in records:
        thread = record.get("thread", {})
        article = RawArticle(
            article_id=record["uuid"],
            source_type="external_dataset",
            source_name=thread.get("site", "unknown"),
            url=record.get("url") or thread.get("url", ""),
            title=record.get("title") or thread.get("title", ""),
            body=record.get("text", ""),
            published_at=record.get("published") or thread.get("published", ""),
        )
        cleaned.append(cleaner.process(article))

    assert all(item.cleaned_text for item in cleaned)
    assert {item.source_name for item in cleaned} >= {
        "reuters.com",
        "cnbc.com",
    }
