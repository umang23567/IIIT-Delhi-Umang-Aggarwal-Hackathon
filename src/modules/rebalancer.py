"""Module A: tactical, contagion-aware index rebalancer with turnover cap + backtest."""
import numpy as np
import pandas as pd


def rebalance(S: pd.DataFrame, k: float = 0.35, lo: float = 0.4, hi: float = 2.0,
              turnover_cap: float = 0.15) -> tuple[pd.DataFrame, pd.Series]:
    """Tilt equal-weight base by signal; clamp to [lo,hi]x base; cap one-way turnover per step."""
    n = S.shape[1]
    base = np.full(n, 1.0 / n)
    scale = S.values.std() + 1e-9
    prev, rows, turns = base.copy(), [], []
    for _, row in S.iterrows():
        z = np.clip((row.values - row.values.mean()) / scale, -2.5, 2.5)
        w = base * np.exp(k * z)
        for _ in range(8):
            w = np.clip(w, lo * base, hi * base)
            w = w / w.sum()
        move = w - prev
        t = 0.5 * np.abs(move).sum()
        if t > turnover_cap:
            move *= turnover_cap / t
            t = turnover_cap
        w = prev + move
        rows.append(w)
        turns.append(t)
        prev = w
    return pd.DataFrame(rows, index=S.index, columns=S.columns), pd.Series(turns, index=S.index)


def backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Weights set at close of day t earn returns of day t+1."""
    return (weights.shift(1) * returns).sum(axis=1).iloc[1:]


def metrics(ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    cum = (1 + ret).cumprod()
    vol = ret.std() * np.sqrt(252)
    return dict(cum_return=round(float(cum.iloc[-1] - 1), 4),
                ann_sharpe=round(float(ret.mean() * 252 / vol), 2) if vol > 0 else 0.0,
                max_drawdown=round(float((cum / cum.cummax() - 1).min()), 4),
                avg_turnover=round(float(turnover.mean()), 4) if turnover is not None else 0.0)
