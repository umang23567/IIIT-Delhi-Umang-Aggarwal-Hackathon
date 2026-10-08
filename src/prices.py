"""Price loading. Default: data/prices.csv. Optional: real prices via yfinance."""
import pandas as pd

from .config import DATA, TICKERS


def load_prices() -> pd.DataFrame:
    p = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
    return p.pivot(index="date", columns="ticker", values="close")[TICKERS]


def load_returns() -> pd.DataFrame:
    return load_prices().pct_change().fillna(0.0)


def fetch_yfinance(start: str, end: str) -> None:
    """Overwrite data/prices.csv with real prices (needs internet + `pip install yfinance`)."""
    import yfinance as yf
    px = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    px.columns = ["date", "ticker", "close"]
    px.dropna().to_csv(DATA / "prices.csv", index=False)
