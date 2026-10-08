"""Module D: explainable daily risk brief (template-based, every claim cites its evidence)."""
import pandas as pd


def make_brief(sig: pd.DataFrame, triggers: pd.DataFrame, blocked: pd.DataFrame,
               stress: list, weights: pd.DataFrame, validation: dict) -> str:
    last = weights.index[-1]
    prev = weights.index[max(0, len(weights) - 6)]
    delta = (weights.loc[last] - weights.loc[prev]).sort_values()
    L = [f"# ContagionLens Risk Brief - {last.date()}", "",
         f"Signals processed: **{len(sig)}** from **{sig.source.nunique()}** sources "
         f"({int(sig.uncorroborated.sum())} flagged uncorroborated).", "", "## Top adverse events"]
    for r in triggers.head(5).itertuples():
        L.append(f"- **{r.event_type}** on **{r.entity}** (impact {r.impact}, sentiment {r.sentiment:+.2f}, "
                 f"cred {r.credibility}) - \"{r.text[:110]}\"")
    if not blocked.empty:
        L += ["", "## Blocked from triggering stress tests (rumour filter)"]
        for r in blocked.head(3).itertuples():
            L.append(f"- {r.entity}: impact {r.impact} but **{r.reason}** - \"{r.text[:90]}\"")
    L += ["", "## Portfolio stress results"]
    for s in stress[:3]:
        L.append(f"- {s['event_type']} / {s['entity']}: book {s['value_before']/1e6:,.1f}M -> "
                 f"{s['value_after']/1e6:,.1f}M ({s['pnl_pct']:+.2f}%)")
    L += ["", "## Index tilt (5-day change in weight)",
          "- Biggest cuts: " + ", ".join(f"{t} ({v*100:+.2f}pp)" for t, v in delta.head(3).items()),
          "- Biggest adds: " + ", ".join(f"{t} ({v*100:+.2f}pp)" for t, v in delta.tail(3)[::-1].items()),
          "", "## Model health"]
    q = validation.get("nlp_quality", {}).get("all", {})
    L.append(f"- Event accuracy {q.get('event_acc')}, sentiment sign accuracy {q.get('sentiment_sign_acc')}")
    return "\n".join(L) + "\n"
