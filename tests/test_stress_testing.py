from src.stress.stress_engine import WholesalePortfolioStressEngine


def test_accepted_stress_test_calculates_loss_and_attribution():
    result = WholesalePortfolioStressEngine().run_stress_test(
        signal_title="Boeing Debt Downgraded",
        event_type="Credit Event",
        impact_score=8.5,
        confidence=0.88,
        source_name="Reuters",
        affected_ticker="BA",
        corroboration_count=2,
    )

    assert result.policy_decision.accepted is True
    assert result.total_pnl_loss > 0.0
    assert result.total_stressed_portfolio_value < result.total_baseline_portfolio_value
    assert result.issuer_losses["The Boeing Company"] > 0.0
    assert result.sector_losses["Industrials"] > 0.0


def test_rejected_stress_test_does_not_execute():
    result = WholesalePortfolioStressEngine().run_stress_test(
        signal_title="Non-material rumor",
        event_type="Credit Event",
        impact_score=6.9,
        confidence=0.90,
        source_name="Reuters",
    )

    assert result.policy_decision.accepted is False
    assert result.position_impacts == []
    assert result.total_pnl_loss == 0.0


def test_single_source_stress_requires_corroboration():
    result = WholesalePortfolioStressEngine().run_stress_test(
        signal_title="Unconfirmed credit rumor",
        event_type="Credit Event",
        impact_score=8.5,
        confidence=0.88,
        source_name="Reuters",
    )

    assert result.policy_decision.accepted is False
    assert "corroborates" in result.policy_decision.trigger_reason
