from src.pipeline.signal_store import RiskSignalStore, StructuredRiskSignal


def test_signal_corroboration_clusters_independent_sources(tmp_path):
    store = RiskSignalStore(str(tmp_path / "signals.db"))
    common = dict(
        published_at="2026-10-09T10:00:00Z",
        source_type="News",
        ticker="BA",
        company_name="The Boeing Company",
        sector="Industrials",
        sentiment_score=-0.8,
        sentiment_label="NEGATIVE",
        sentiment_probs={"positive": 0.1, "neutral": 0.1, "negative": 0.8},
        event_type="Credit Event",
        event_confidence=0.9,
        evidence_text="credit downgrade",
        impact_score=8.5,
        impact_confidence=0.8,
        impact_drivers=[],
        materiality="CRITICAL",
        model_version="test",
    )
    store.save_signal(StructuredRiskSignal(
        signal_id="SIG_A", article_id="A", source_name="Reuters",
        title="Boeing debt downgraded after liquidity strain", url="https://a", **common
    ))
    store.save_signal(StructuredRiskSignal(
        signal_id="SIG_B", article_id="B", source_name="Bloomberg",
        title="Boeing credit rating cut as liquidity strain deepens", url="https://b", **common
    ))
    store.recompute_corroboration()
    signals = {signal.signal_id: signal for signal in store.get_all_signals()}

    assert signals["SIG_A"].corroboration_count == 2
    assert set(signals["SIG_A"].corroborating_sources) == {"Reuters", "Bloomberg"}
