"""Fetch REAL data into ./data (run this on your own machine with internet access).

  pip install yfinance requests pandas
  python scripts/fetch_data.py --start 2025-06-01 --end 2026-09-30
  python scripts/fetch_data.py --tweets path/to/kaggle_tweets.csv --text-col tweet --time-col date

Outputs (same schema the pipeline already reads):
  data/prices.csv  : date,ticker,close
  data/news.csv    : doc_id,timestamp,text,followers
  data/tweets.csv  : doc_id,timestamp,text,followers      (only if --tweets given)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATA, UNIVERSE, TICKERS  # noqa: E402


def fetch_prices(start, end):
    import yfinance as yf
    px = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    long = px.reset_index().melt(id_vars=px.index.name or "Date", var_name="ticker", value_name="close")
    long.columns = ["date", "ticker", "close"]
    long = long.dropna()
    long.to_csv(DATA / "prices.csv", index=False)
    print(f"prices: {len(long)} rows, {long.ticker.nunique()} tickers, {long.date.min()} -> {long.date.max()}")


def fetch_gdelt(start, end, per_company=250):
    """GDELT DOC 2.0 article list (titles only, free, no key). Chunked by month to dodge the 250 cap."""
    import requests
    rows = []
    months = pd.date_range(start, end, freq="MS")
    for t, (name, _, _) in UNIVERSE.items():
        for m in months:
            s, e = m, min(m + pd.offsets.MonthEnd(0), pd.Timestamp(end))
            q = f'"{name}" sourcelang:english'
            url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + requests.utils.quote(q) +
                   f"&mode=artlist&maxrecords={per_company}&format=json&sort=datedesc"
                   f"&startdatetime={s:%Y%m%d}000000&enddatetime={e:%Y%m%d}235959")
            try:
                r = requests.get(url, timeout=30)
                arts = r.json().get("articles", []) if r.text.strip().startswith("{") else []
            except Exception as ex:
                print("skip", t, m.date(), ex)
                arts = []
            for a in arts:
                rows.append(dict(timestamp=pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce"),
                                 text=a["title"], followers=None))
            time.sleep(5.5)  # GDELT asks for ~1 request / 5 s
        print(f"{t}: running total {len(rows)}")
    df = pd.DataFrame(rows).dropna(subset=["timestamp", "text"]).drop_duplicates("text")
    df.insert(0, "doc_id", [f"n{i}" for i in range(1, len(df) + 1)])
    df.to_csv(DATA / "news.csv", index=False)
    print(f"news: {len(df)} headlines")


def adapt_csv(path, out_name, text_col, time_col, followers_col=None, prefix="t"):
    """Map any Kaggle CSV to the pipeline schema, with clear errors on missing columns."""
    df = pd.read_csv(path)
    for c in [text_col, time_col] + ([followers_col] if followers_col else []):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not in {path}. Available: {list(df.columns)}")
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_localize(None),
                        "text": df[text_col].astype(str),
                        "followers": df[followers_col] if followers_col else None}).dropna(subset=["timestamp"])
    out.insert(0, "doc_id", [f"{prefix}{i}" for i in range(1, len(out) + 1)])
    out.to_csv(DATA / out_name, index=False)
    print(f"{out_name}: {len(out)} rows, {out.timestamp.min()} -> {out.timestamp.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--tweets", help="Kaggle tweets CSV to adapt")
    ap.add_argument("--news-csv", help="Kaggle news CSV to adapt instead of GDELT")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--followers-col")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if not a.skip_prices:
        fetch_prices(a.start, a.end)
    if a.news_csv:
        adapt_csv(a.news_csv, "news.csv", a.text_col, a.time_col, prefix="n")
    elif not a.skip_news:
        fetch_gdelt(a.start, a.end)
    if a.tweets:
        adapt_csv(a.tweets, "tweets.csv", a.text_col, a.time_col, a.followers_col)
