"""End-to-end orchestration: engine -> contagion -> modules A-D -> output/ files."""
import json

import pandas as pd

from . import config as C
from .data_gen import generate
from .engine import run as run_engine, daily_scores
from .graph import apply_contagion, edge_table
from .prices import load_returns
from .modules.rebalancer import rebalance, backtest, metrics
from .modules.stress import detect_triggers, run_stress
from .modules.validator import information_coefficient, nlp_quality
from .modules.brief import make_brief


def run_all(regen: bool = False, max_stress: int = 5) -> dict:
    C.OUT.mkdir(exist_ok=True, parents=True)
    if regen or not (C.DATA / "news.csv").exists():
        print("Generating synthetic data ->", generate())
    R = load_returns()
    docs, sig = run_engine("engine")
    docs_n, sig_n = run_engine("naive")

    S = daily_scores(sig, R.index)
    S_c = apply_contagion(S)
    S_n = daily_scores(sig_n, R.index)

    # Module A: rebalancer variants + backtest
    variants = {"ContagionLens (graph)": S_c, "Engine, no graph": S, "Naive NLP baseline": S_n}
    bt, bt_metrics, weights = {}, {}, {}
    for name, sc in variants.items():
        w, turn = rebalance(sc)
        bt[name] = backtest(w, R)
        bt_metrics[name] = metrics(bt[name], turn)
        weights[name] = w
    ew = pd.DataFrame(1 / len(C.TICKERS), index=R.index, columns=C.TICKERS)
    bt["Equal-weight benchmark"] = backtest(ew, R)
    bt_metrics["Equal-weight benchmark"] = metrics(bt["Equal-weight benchmark"])
    w_main = weights["ContagionLens (graph)"]
    w_main.to_csv(C.OUT / "weights.csv")
    pd.DataFrame(bt).to_csv(C.OUT / "backtest.csv")
    S_c.to_csv(C.OUT / "scores_contagion.csv"); S.to_csv(C.OUT / "scores_raw.csv")
    edge_table(C.TICKERS).to_csv(C.OUT / "contagion_edges.csv", index=False)

    # Module B: stress tests on triggers
    pf = pd.read_csv(C.DATA / "portfolio.csv")
    trig, blocked = detect_triggers(sig)
    diverse = trig.groupby("event_type").head(1).sort_values("impact", ascending=False)  # one per event type
    stress = [run_stress(pf, r.event_type, r.impact, r.entity, r.scope, r.sentiment)
              for r in diverse.head(max_stress).itertuples()]
    (C.OUT / "stress_results.json").write_text(json.dumps(stress, indent=1, default=str))
    trig.to_csv(C.OUT / "stress_triggers.csv", index=False)
    blocked.to_csv(C.OUT / "stress_blocked.csv", index=False)

    # Module C: validation
    validation = dict(
        ic=dict(contagion=information_coefficient(S_c, R), engine_no_graph=information_coefficient(S, R),
                naive=information_coefficient(S_n, R)),
        nlp_quality=nlp_quality(docs), nlp_quality_naive=nlp_quality(docs_n),
        backtest=bt_metrics,
        rumour_filter=dict(blocked_from_stress=int(len(blocked)),
                           rumours_flagged=int(sig[(sig.is_rumor == True) & sig.uncorroborated].shape[0]),
                           rumour_signals_total=int((sig.is_rumor == True).sum())))
    (C.OUT / "validation.json").write_text(json.dumps(validation, indent=1))

    # Module D: brief
    (C.OUT / "brief.md").write_text(make_brief(sig, trig, blocked, stress, w_main, validation))
    return validation


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1))
