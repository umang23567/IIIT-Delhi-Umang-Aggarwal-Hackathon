"""Module C: signal validator. Measures predictive power and NLP quality vs a naive baseline."""
import numpy as np
import pandas as pd

from ..config import EVENT_TYPES


def information_coefficient(S: pd.DataFrame, R: pd.DataFrame) -> dict:
    """Daily cross-sectional Spearman rank-IC of signal(t) vs next-day return."""
    fwd = R.shift(-1)
    ics = []
    for d in S.index[:-1]:
        s, r = S.loc[d], fwd.loc[d]
        if s.std() > 1e-12:
            ics.append(s.rank().corr(r.rank()))
    ics = pd.Series(ics).dropna()
    n = len(ics)
    mean = float(ics.mean()) if n else 0.0
    t = mean / (ics.std() / np.sqrt(n)) if n > 2 and ics.std() > 0 else 0.0
    return dict(mean_ic=round(mean, 4), t_stat=round(float(t), 2), hit_rate=round(float((ics > 0).mean()), 3) if n else 0.0, n_days=n)


def macro_f1(y_true, y_pred) -> float:
    f1s = []
    for c in EVENT_TYPES:
        tp = sum((a == c) and (b == c) for a, b in zip(y_true, y_pred))
        fp = sum((a != c) and (b == c) for a, b in zip(y_true, y_pred))
        fn = sum((a == c) and (b != c) for a, b in zip(y_true, y_pred))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def nlp_quality(docs: pd.DataFrame) -> dict:
    """Needs ground-truth columns true_event / true_sentiment (synthetic or hand-labelled)."""
    d = docs.dropna(subset=["true_event"]).copy()
    pred_sign = np.where(d.sentiment > 0.1, 1, np.where(d.sentiment < -0.1, -1, 0))
    out = {}
    for name, g in [("all", d), ("news", d[d.source == "news"]), ("twitter", d[d.source == "twitter"])]:
        ps = pred_sign[g.index.map(d.index.get_loc)]
        out[name] = dict(sentiment_sign_acc=round(float((ps == g.true_sentiment.values).mean()), 3),
                         event_acc=round(float((g.event_type == g.true_event).mean()), 3),
                         event_macro_f1=round(macro_f1(g.true_event.tolist(), g.event_type.tolist()), 3), n=len(g))
    return out
