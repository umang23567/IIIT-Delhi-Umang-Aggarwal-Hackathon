"""Contagion graph: how an event on one name leaks to connected names."""
import numpy as np
import pandas as pd

from .config import (MANUAL_EDGES, SECTOR_PEER_WEIGHT, UNIVERSE, CONTAGION_LAMBDA,
                     NEG_AMPLIFY, POS_DAMPEN)


def edge_table(tickers) -> pd.DataFrame:
    """Directed edges incl. default sector-peer links. Manual edges override defaults."""
    edges = {(s, t): w for s, t, w in MANUAL_EDGES if s in tickers and t in tickers}
    for a in tickers:
        for b in tickers:
            if a != b and UNIVERSE[a][1] == UNIVERSE[b][1] and (a, b) not in edges:
                edges[(a, b)] = SECTOR_PEER_WEIGHT
    return pd.DataFrame([(s, t, w) for (s, t), w in edges.items()],
                        columns=["source", "target", "weight"])


def build_adjacency(tickers) -> np.ndarray:
    """A[i, j] = influence of j's risk on i."""
    idx = {t: i for i, t in enumerate(tickers)}
    A = np.zeros((len(tickers), len(tickers)))
    for _, r in edge_table(tickers).iterrows():
        A[idx[r.target], idx[r.source]] = r.weight
    return A


def apply_contagion(S: pd.DataFrame, lam: float = CONTAGION_LAMBDA, hops: int = 2) -> pd.DataFrame:
    """Spread signals along the graph. Negative signals are amplified, positive dampened."""
    A = build_adjacency(list(S.columns))
    g = np.where(S.values < 0, S.values * NEG_AMPLIFY, S.values * POS_DAMPEN)
    out = S.values.copy()
    cur = g
    for h in range(1, hops + 1):
        cur = cur @ A.T
        out = out + (lam ** h) * cur
    return pd.DataFrame(out, index=S.index, columns=S.columns)
