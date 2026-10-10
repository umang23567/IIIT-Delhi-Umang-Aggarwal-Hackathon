import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class StressPolicyDecision(BaseModel):
    accepted: bool
    trigger_reason: str
    scenario_id: str
    policy_version: str = "BASEL_POLICY_2026_V1"

class PositionStressImpact(BaseModel):
    position_id: str
    asset_type: str
    ticker: Optional[str]
    issuer: str
    sector: str
    baseline_value: float
    stressed_value: float
    pnl_loss: float
    percentage_change: float
    applied_shock: str

class StressTestResult(BaseModel):
    run_id: str
    event_title: str
    event_type: str
    impact_score: float
    scenario_name: str
    policy_decision: StressPolicyDecision
    total_baseline_portfolio_value: float
    total_stressed_portfolio_value: float
    total_pnl_loss: float
    total_percentage_loss: float
    asset_class_losses: Dict[str, float]
    issuer_losses: Dict[str, float]
    sector_losses: Dict[str, float]
    position_impacts: List[PositionStressImpact]
    assumptions: List[str]

class WholesalePortfolioStressEngine:
    """
    Module B: Strategic Portfolio Stress Testing on Wholesale Banking Assets.
    Simulates multi-asset shocks on:
    - Equities: P&L = Market Value * Beta * Market Return Shock
    - Corporate Bonds: Price Change = -Duration * Yield Shock + Credit Spread Shock
    - Interest Rate Swaps: P&L = -DV01 * Rate Change
    - Wholesale Loans: Loss = EAD * (Stressed PD * Stressed LGD - Base PD * Base LGD)
    """
    def __init__(self, portfolio_path: str = "data/portfolios/synthetic_portfolio.json"):
        with open(portfolio_path, "r", encoding="utf-8") as f:
            self.portfolio_data = json.load(f)
        self.positions = self.portfolio_data["positions"]

    def evaluate_trigger(
        self,
        impact_score: float,
        confidence: float,
        source_type: str,
        corroboration_count: int = 1,
        novelty_score: float = 1.0,
    ) -> StressPolicyDecision:
        """Policy gate evaluating if a signal warrants a formal stress test."""
        # Formal stress testing requires materiality and model confidence.
        if impact_score < 7.0:
            return StressPolicyDecision(
                accepted=False,
                trigger_reason=f"Impact score {impact_score:.1f} below systemic threshold of 7.0 (Non-material noise)",
                scenario_id="NO_TRIGGER"
            )
        if confidence < 0.65:
            return StressPolicyDecision(
                accepted=False,
                trigger_reason=f"Model confidence {confidence:.2f} below risk threshold 0.65 (Needs human verification)",
                scenario_id="NO_TRIGGER"
            )
        if corroboration_count < 2:
            return StressPolicyDecision(
                accepted=False,
                trigger_reason="Only one independent source corroborates this event; human verification required.",
                scenario_id="NO_TRIGGER"
            )
        return StressPolicyDecision(
            accepted=True,
            trigger_reason=f"High impact signal ({impact_score:.1f}/10, conf: {confidence:.2f}) verified. Policy gate accepted.",
            scenario_id=f"STRESS_SCENARIO_{int(impact_score*10)}"
        )

    def run_stress_test(
        self,
        signal_title: str,
        event_type: str,
        impact_score: float,
        confidence: float,
        source_name: str,
        affected_ticker: Optional[str] = None,
        custom_positions: Optional[List[Dict[str, Any]]] = None,
        corroboration_count: int = 1,
        novelty_score: float = 1.0,
    ) -> StressTestResult:
        policy = self.evaluate_trigger(
            impact_score,
            confidence,
            source_name,
            corroboration_count=corroboration_count,
            novelty_score=novelty_score,
        )
        if not policy.accepted:
            return StressTestResult(
                run_id=f"RUN_{abs(hash(signal_title)) % 1000000}",
                event_title=signal_title,
                event_type=event_type,
                impact_score=impact_score,
                scenario_name="Not executed - policy gate rejected",
                policy_decision=policy,
                total_baseline_portfolio_value=0.0,
                total_stressed_portfolio_value=0.0,
                total_pnl_loss=0.0,
                total_percentage_loss=0.0,
                asset_class_losses={"Equity": 0.0, "Bond": 0.0, "Loan": 0.0, "Swap": 0.0},
                issuer_losses={},
                sector_losses={},
                position_impacts=[],
                assumptions=[policy.trigger_reason],
            )

        # Determine calibrated shock parameters based on event type
        shocks = self._map_event_to_shocks(event_type, impact_score, affected_ticker)

        positions_to_eval = custom_positions if custom_positions is not None else self.positions
        total_base = 0.0
        total_stress = 0.0
        position_results: List[PositionStressImpact] = []
        asset_losses = {"Equity": 0.0, "Bond": 0.0, "Loan": 0.0, "Swap": 0.0}
        issuer_losses: Dict[str, float] = {}
        sector_losses: Dict[str, float] = {}

        for p in positions_to_eval:
            a_type = p["asset_type"]
            ticker = p.get("ticker")

            if a_type == "Equity":
                mv = p["market_value"]
                beta = p["beta"]
                total_base += mv
                
                # Check if this company is directly targeted
                if affected_ticker and ticker == affected_ticker:
                    ret_shock = shocks["single_equity_drop"]
                else:
                    ret_shock = shocks["market_equity_drop"]

                loss = mv * beta * ret_shock
                stressed_mv = max(0.0, mv - loss)
                total_stress += stressed_mv
                asset_losses["Equity"] += loss
                issuer_losses[p["issuer"]] = issuer_losses.get(p["issuer"], 0.0) + loss
                sector_losses[p["sector"]] = sector_losses.get(p["sector"], 0.0) + loss

                pct = (-loss / mv) * 100.0 if mv > 0 else 0.0
                position_results.append(PositionStressImpact(
                    position_id=p["position_id"],
                    asset_type=a_type,
                    ticker=ticker,
                    issuer=p["issuer"],
                    sector=p["sector"],
                    baseline_value=mv,
                    stressed_value=stressed_mv,
                    pnl_loss=round(loss, 2),
                    percentage_change=round(pct, 2),
                    applied_shock=f"Equity Drop: {ret_shock*100:.1f}%, Beta: {beta}"
                ))

            elif a_type == "Bond":
                mv = p["market_value"]
                duration = p["duration"]
                total_base += mv

                # Rate shift + credit spread widening
                rate_shock_bps = shocks["rate_shock_bps"]
                spread_widening_bps = shocks["single_bond_spread_bps"] if (affected_ticker and ticker == affected_ticker) else shocks["credit_spread_bps"]

                # Price change % ≈ -Duration * (Delta y + Delta spread)
                total_yield_shift = (rate_shock_bps + spread_widening_bps) / 10000.0
                price_drop_pct = duration * total_yield_shift
                loss = mv * price_drop_pct
                stressed_mv = max(0.0, mv - loss)
                total_stress += stressed_mv
                asset_losses["Bond"] += loss
                issuer_losses[p["issuer"]] = issuer_losses.get(p["issuer"], 0.0) + loss
                sector_losses[p["sector"]] = sector_losses.get(p["sector"], 0.0) + loss

                pct = (-loss / mv) * 100.0 if mv > 0 else 0.0
                position_results.append(PositionStressImpact(
                    position_id=p["position_id"],
                    asset_type=a_type,
                    ticker=ticker,
                    issuer=p["issuer"],
                    sector=p["sector"],
                    baseline_value=mv,
                    stressed_value=stressed_mv,
                    pnl_loss=round(loss, 2),
                    percentage_change=round(pct, 2),
                    applied_shock=f"-Dur({duration}) x Yield({rate_shock_bps}bps) + Spread({spread_widening_bps}bps)"
                ))

            elif a_type == "Loan":
                ead = p["ead"]
                base_pd = p["pd"]
                base_lgd = p["lgd"]
                # Basel Expected Loss = EAD * PD * LGD
                base_el = ead * base_pd * base_lgd
                total_base += ead # Book exposure

                # Stress PD and LGD
                pd_mult = shocks["single_loan_pd_multiplier"] if (affected_ticker and ticker == affected_ticker) else shocks["loan_pd_multiplier"]
                stressed_pd = min(1.0, base_pd * pd_mult)
                stressed_lgd = min(1.0, base_lgd * 1.15) # 15% recovery stress

                stressed_el = ead * stressed_pd * stressed_lgd
                loss = stressed_el - base_el # incremental expected loss provision
                stressed_value = ead - loss
                total_stress += stressed_value
                asset_losses["Loan"] += loss
                issuer_losses[p["issuer"]] = issuer_losses.get(p["issuer"], 0.0) + loss
                sector_losses[p["sector"]] = sector_losses.get(p["sector"], 0.0) + loss

                pct = (-loss / ead) * 100.0 if ead > 0 else 0.0
                position_results.append(PositionStressImpact(
                    position_id=p["position_id"],
                    asset_type=a_type,
                    ticker=ticker,
                    issuer=p["issuer"],
                    sector=p["sector"],
                    baseline_value=ead,
                    stressed_value=stressed_value,
                    pnl_loss=round(loss, 2),
                    percentage_change=round(pct, 2),
                    applied_shock=f"PD: {base_pd*100:.1f}%->{stressed_pd*100:.1f}%, LGD: {base_lgd*100:.1f}%->{stressed_lgd*100:.1f}%"
                ))

            elif a_type == "Swap":
                notional = p["notional"]
                dv01 = p["dv01"]
                # Book value approximation for swap baseline
                total_base += notional * 0.02 # Net market value estimate
                
                rate_change_bps = shocks["rate_shock_bps"]
                # Swap P&L = -DV01 * rate_change_bps
                loss = dv01 * rate_change_bps # pay/receive convention embedded in dv01 sign
                stressed_mv = (notional * 0.02) - loss
                total_stress += stressed_mv
                asset_losses["Swap"] += loss
                issuer_losses[p["issuer"]] = issuer_losses.get(p["issuer"], 0.0) + loss
                sector_losses[p["sector"]] = sector_losses.get(p["sector"], 0.0) + loss

                pct = (-loss / (notional * 0.02)) * 100.0 if notional > 0 else 0.0
                position_results.append(PositionStressImpact(
                    position_id=p["position_id"],
                    asset_type=a_type,
                    ticker=ticker,
                    issuer=p["issuer"],
                    sector=p["sector"],
                    baseline_value=notional * 0.02,
                    stressed_value=stressed_mv,
                    pnl_loss=round(loss, 2),
                    percentage_change=round(pct, 2),
                    applied_shock=f"DV01(${dv01:,.0f}) x RateShift({rate_change_bps}bps)"
                ))

        total_loss = sum(asset_losses.values())
        pct_loss = (-total_loss / total_base) * 100.0 if total_base > 0 else 0.0

        return StressTestResult(
            run_id=f"RUN_{abs(hash(signal_title)) % 1000000}",
            event_title=signal_title,
            event_type=event_type,
            impact_score=impact_score,
            scenario_name=shocks["scenario_name"],
            policy_decision=policy,
            total_baseline_portfolio_value=round(total_base, 2),
            total_stressed_portfolio_value=round(total_stress, 2),
            total_pnl_loss=round(total_loss, 2),
            total_percentage_loss=round(pct_loss, 2),
            asset_class_losses={k: round(v, 2) for k, v in asset_losses.items()},
            issuer_losses={k: round(v, 2) for k, v in issuer_losses.items() if v > 0},
            sector_losses={k: round(v, 2) for k, v in sector_losses.items() if v > 0},
            position_impacts=position_results,
            assumptions=shocks["assumptions"]
        )

    def _map_event_to_shocks(self, event_type: str, impact_score: float, affected_ticker: Optional[str]) -> Dict[str, Any]:
        """Maps an event classification to consistent multi-factor macro/credit shocks."""
        severity_ratio = impact_score / 10.0

        if event_type == "Credit Event":
            return {
                "scenario_name": "Credit Distress & Rating Migration Shock",
                "market_equity_drop": 0.03 * severity_ratio,
                "single_equity_drop": 0.18 * severity_ratio,
                "rate_shock_bps": -15.0 * severity_ratio, # flight to quality
                "credit_spread_bps": 40.0 * severity_ratio,
                "single_bond_spread_bps": 220.0 * severity_ratio,
                "loan_pd_multiplier": 1.20,
                "single_loan_pd_multiplier": 2.50,
                "assumptions": [
                    "Targeted issuer credit downgrade widens single-name CDS/spread by up to 220bps.",
                    "Issuer loan Default Probability (PD) increases 2.5x under distressed liquidity strain.",
                    "Mild flight-to-safety rally compresses benchmark sovereign yields by 15bps."
                ]
            }
        elif event_type == "Geopolitical":
            return {
                "scenario_name": "Global Maritime Chokepoint & Energy Disruption",
                "market_equity_drop": 0.09 * severity_ratio,
                "single_equity_drop": 0.09 * severity_ratio,
                "rate_shock_bps": 35.0 * severity_ratio, # stagflationary rate surge
                "credit_spread_bps": 95.0 * severity_ratio,
                "single_bond_spread_bps": 95.0 * severity_ratio,
                "loan_pd_multiplier": 1.45,
                "single_loan_pd_multiplier": 1.45,
                "assumptions": [
                    "Broad equity market contraction of up to 9% due to global supply shocks.",
                    "Energy spike adds inflationary pressure, driving yields up 35bps.",
                    "Wholesale credit spreads widen 95bps across cyclical and industrial sectors."
                ]
            }
        elif event_type == "Macroeconomic":
            return {
                "scenario_name": "Central Bank Emergency Policy Tightening",
                "market_equity_drop": 0.08 * severity_ratio,
                "single_equity_drop": 0.08 * severity_ratio,
                "rate_shock_bps": 50.0 * severity_ratio, # 50bps rate jump
                "credit_spread_bps": 60.0 * severity_ratio,
                "single_bond_spread_bps": 60.0 * severity_ratio,
                "loan_pd_multiplier": 1.35,
                "single_loan_pd_multiplier": 1.35,
                "assumptions": [
                    "Yield curve parallel shift upward by 50bps.",
                    "Duration drag leads to immediate bond price depreciation.",
                    "Interest rate swap receive-positions face marked negative DV01 loss."
                ]
            }
        elif event_type == "Regulatory/Legal":
            return {
                "scenario_name": "Regulatory Enforcement & Capital Surcharge Shock",
                "market_equity_drop": 0.02 * severity_ratio,
                "single_equity_drop": 0.12 * severity_ratio,
                "rate_shock_bps": 0.0,
                "credit_spread_bps": 20.0 * severity_ratio,
                "single_bond_spread_bps": 120.0 * severity_ratio,
                "loan_pd_multiplier": 1.10,
                "single_loan_pd_multiplier": 1.80,
                "assumptions": [
                    "Targeted regulatory consent order prompts compliance capital deduction.",
                    "Affected bank/issuer equity drops 12% on revised dividend restrictions."
                ]
            }
        else:
            # Baseline shock for moderate events
            return {
                "scenario_name": "Moderate Operational/Market Sentiment Shock",
                "market_equity_drop": 0.02 * severity_ratio,
                "single_equity_drop": 0.04 * severity_ratio,
                "rate_shock_bps": 5.0 * severity_ratio,
                "credit_spread_bps": 15.0 * severity_ratio,
                "single_bond_spread_bps": 25.0 * severity_ratio,
                "loan_pd_multiplier": 1.05,
                "single_loan_pd_multiplier": 1.10,
                "assumptions": [
                    "Idiosyncratic headline creates minor market fluctuation without systemic contagion."
                ]
            }
