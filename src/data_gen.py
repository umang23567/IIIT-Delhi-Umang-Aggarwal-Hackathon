"""Synthetic data generator: news, tweets, prices, portfolio.

IMPORTANT: prices contain a PLANTED event->return effect (incl. graph spillover) so the
pipeline can be demonstrated offline. Validation on this data is a sanity check of the
plumbing, NOT evidence of real alpha. Re-run validation on real prices (see prices.py).
"""
import numpy as np
import pandas as pd

from .config import DATA, UNIVERSE, TICKERS
from .graph import build_adjacency

SEV = {"Credit Event": 0.9, "Geopolitical": 0.8, "Macroeconomic": 0.7, "Cyber/Operational": 0.7,
       "Regulatory/Legal": 0.6, "Earnings": 0.6, "Merger/Acquisition": 0.5, "Product Launch": 0.4}

COMPANY_NEWS = {
    "Credit Event": {
        -1: ["{c} downgraded by rating agency amid mounting debt concerns",
             "{c} misses bond payment, default fears grow",
             "{c} credit spreads widen sharply as lenders cut exposure"],
        1: ["{c} upgraded as balance sheet strengthens and leverage falls"]},
    "Merger/Acquisition": {
        1: ["{c} agrees to acquire rival in ${amt} billion deal, shares surge"],
        -1: ["{c} merger blocked, deal collapses after regulatory pushback"]},
    "Product Launch": {
        1: ["{c} unveils new product line, analysts upbeat on strong demand"],
        -1: ["{c} delays product launch after supply problems"]},
    "Earnings": {
        1: ["{c} beats earnings expectations, raises guidance on record revenue"],
        -1: ["{c} profit falls short, cuts outlook as margins shrink"]},
    "Regulatory/Legal": {
        -1: ["{c} hit with ${amt} billion fine in antitrust probe",
             "{c} faces lawsuit and investigation over alleged violations"],
        1: ["{c} wins appeal as regulator drops probe"]},
    "Cyber/Operational": {
        -1: ["{c} confirms data breach after cyberattack, operations disrupted",
             "{c} halts production after plant fire"],
        1: ["{c} restores operations, strong recovery after outage"]},
}
MARKET_NEWS = [
    ("Geopolitical", -1, "Escalating conflict and sanctions rattle global markets as tensions rise"),
    ("Geopolitical", -1, "Trade war fears grow as tariffs imposed on imports"),
    ("Geopolitical", 1, "Ceasefire agreement eases tensions, markets rally"),
    ("Macroeconomic", -1, "Central bank raises rates by {pct} points, inflation fears weigh on stocks"),
    ("Macroeconomic", -1, "Recession fears mount as unemployment rises"),
    ("Macroeconomic", 1, "Inflation cools, rate cut hopes lift markets"),
]
SECTOR_NEWS = [
    ("Geopolitical", 1, "Energy", "Sanctions disrupt supply, oil prices surge as producers rally"),
    ("Regulatory/Legal", -1, "Financials", "Banks face tougher capital rules, banking stocks slide"),
    ("Regulatory/Legal", -1, "Technology", "New export curbs on chip sales hit tech stocks, semiconductor shares slide"),
    ("Regulatory/Legal", 1, "Healthcare", "FDA fast-tracks drug approvals, pharma stocks climb"),
    ("Macroeconomic", -1, "Consumer", "Weak consumer spending data hits retail shares"),
]
TWEET_COMPANY = {
    "Credit Event": {-1: ["{t} downgrade incoming?? looks rough 😬", "{t} default fears are real, lenders running 📉"],
                     1: ["{t} upgrade is huge 🚀 balance sheet looking strong"]},
    "Merger/Acquisition": {1: ["{t} acquisition news, shares surge 🚀"], -1: ["{t} deal collapses, ugh 📉"]},
    "Product Launch": {1: ["{t} new launch is huge 🚀🚀", "loving the {t} unveils today 📈"],
                       -1: ["{t} delays launch again, ugh 😬"]},
    "Earnings": {1: ["{t} beats earnings, record revenue 📈"], -1: ["{t} earnings miss, margins shrink 📉"]},
    "Regulatory/Legal": {-1: ["{t} fined again lol, lawsuit incoming 😬"], 1: ["{t} wins appeal, probe dropped 📈"]},
    "Cyber/Operational": {-1: ["{t} cyberattack?? breach confirmed 😱"], 1: ["{t} operations restored, strong recovery 📈"]},
}
TWEET_MARKET = {-1: ["markets rattled, tensions and sanctions everywhere 📉", "inflation fears + rate hike = bleeding 📉"],
                1: ["ceasefire news, markets rally 🚀", "inflation cools, rate cut hopes 📈"]}
CHATTER = ["anyone watching {t} today?", "{t} chart looks interesting", "thoughts on {t}?",
           "added some {t} to my watchlist", "{t} volume today is wild"]
RUMOR = ["heard {t} is about to default, sources say 😱", "{t} bankruptcy rumors all over my feed 🚨 lenders pulling out",
         "breaking?? {t} hit by massive cyberattack, operations disrupted 😱"]


def _market_text(t, rng):
    return t.format(pct=rng.choice(["0.25", "0.5", "0.75"]), amt=int(rng.integers(2, 40)))


def generate(seed=7, n_days=45, start="2026-07-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    A = build_adjacency(TICKERS)
    sector_of = {t: UNIVERSE[t][1] for t in TICKERS}
    effects = pd.DataFrame(0.0, index=days, columns=TICKERS)
    news, tweets = [], []
    nid = tid = 0

    def ts(day, h0=8, h1=15):
        return day + pd.Timedelta(hours=int(rng.integers(h0, h1 + 1)), minutes=int(rng.integers(0, 60)))

    def add_tweet(day, text, event, pol, ent, rumor=False, followers=None):
        nonlocal tid
        tid += 1
        f = followers or int(max(50, rng.lognormal(8, 1.5)))
        tweets.append(dict(doc_id=f"t{tid}", timestamp=ts(day, 8, 20), text=text, followers=f,
                           true_event=event, true_sentiment=pol, true_entity=ent, is_rumor=rumor))
        if rng.random() < 0.08:  # retweet duplicate
            tid += 1
            tweets.append(dict(doc_id=f"t{tid}", timestamp=ts(day, 8, 20), text="RT " + text, followers=int(f * 0.2) + 50,
                               true_event=event, true_sentiment=pol, true_entity=ent, is_rumor=rumor))

    for di, day in enumerate(days):
        for _ in range(rng.poisson(6)):
            kind = rng.choice(["company", "sector", "market"], p=[0.7, 0.15, 0.15])
            mag_u = 0.7 + 0.6 * rng.random()
            vec = np.zeros(len(TICKERS))
            if kind == "company":
                et = str(rng.choice(list(COMPANY_NEWS)))
                pol = int(rng.choice([1, -1], p=[0.5, 0.5]))
                tk = str(rng.choice(TICKERS))
                tmpl = str(rng.choice(COMPANY_NEWS[et][pol]))
                text = tmpl.format(c=UNIVERSE[tk][0], amt=int(rng.integers(2, 40)))
                ent = tk
                m = pol * SEV[et] * 0.02 * mag_u
                vec[TICKERS.index(tk)] = m
                spill = (A @ vec) * 0.35 * np.where(pol < 0, 1.3, 1.0)
                vec = vec + spill
                tw = TWEET_COMPANY[et][pol]
                tweet_texts = [str(rng.choice(tw)).format(t="$" + tk) for _ in range(int(rng.integers(0, 4)))]
            elif kind == "sector":
                et, pol, sec, tmpl = SECTOR_NEWS[int(rng.integers(len(SECTOR_NEWS)))]
                text, ent = tmpl, sec
                m = pol * SEV[et] * 0.02 * mag_u
                vec = np.array([m * 0.8 if sector_of[t] == sec else 0.0 for t in TICKERS])
                tweet_texts = []
            else:
                et, pol, tmpl = MARKET_NEWS[int(rng.integers(len(MARKET_NEWS)))]
                text, ent = _market_text(tmpl, rng), "MARKET"
                vec = np.full(len(TICKERS), pol * SEV[et] * 0.02 * mag_u * 0.5)
                tweet_texts = [str(rng.choice(TWEET_MARKET[pol])) for _ in range(int(rng.integers(0, 3)))]
            if di + 1 < len(days):
                effects.iloc[di + 1] += vec
            if rng.random() < 0.85:
                nid += 1
                news.append(dict(doc_id=f"n{nid}", timestamp=ts(day), text=text, followers=np.nan,
                                 true_event=et, true_sentiment=pol, true_entity=ent, is_rumor=False))
            for tt in tweet_texts:
                add_tweet(day, tt, et, pol, ent)
        for _ in range(int(rng.integers(15, 30))):  # chatter
            tk = str(rng.choice(TICKERS))
            add_tweet(day, str(rng.choice(CHATTER)).format(t="$" + tk), "Other", 0, tk)
        if rng.random() < 0.35:  # unverified rumour burst: no news, no price effect
            tk = str(rng.choice(TICKERS))
            base = str(rng.choice(RUMOR)).format(t="$" + tk)
            for _ in range(int(rng.integers(2, 4))):
                add_tweet(day, base, "Credit Event", -1, tk, rumor=True, followers=int(rng.integers(50, 400)))

    # prices
    m = rng.normal(0.0003, 0.008, len(days))
    beta = rng.uniform(0.8, 1.3, len(TICKERS))
    eps = rng.normal(0, 0.009, (len(days), len(TICKERS)))
    rets = pd.DataFrame(m[:, None] * beta + eps + effects.values, index=days, columns=TICKERS)
    rets.iloc[0] = 0.0
    p0 = rng.uniform(50, 400, len(TICKERS))
    prices = (1 + rets).cumprod() * p0
    prices_long = prices.reset_index().melt(id_vars="index", var_name="ticker", value_name="close")
    prices_long.columns = ["date", "ticker", "close"]

    DATA.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(news).to_csv(DATA / "news.csv", index=False)
    pd.DataFrame(tweets).to_csv(DATA / "tweets.csv", index=False)
    prices_long.to_csv(DATA / "prices.csv", index=False)
    make_portfolio(rng).to_csv(DATA / "portfolio.csv", index=False)
    return len(news), len(tweets)


def make_portfolio(rng) -> pd.DataFrame:
    """Synthetic wholesale-banking book: loans, bonds, swaps, options, equities."""
    pd_by_rating = {"AA": 0.0005, "A": 0.001, "BBB": 0.004, "BB": 0.015}
    rows, pid = [], 0

    def row(**k):
        nonlocal pid
        pid += 1
        base = dict(position_id=f"P{pid:03d}", asset_type="", entity="", sector="", notional=0.0,
                    market_value=0.0, rating="", duration=0.0, spread_duration=0.0, dv01=0.0,
                    delta=0.0, pd=0.0, lgd=0.0)
        base.update(k)
        rows.append(base)

    for _ in range(20):
        tk = str(rng.choice(TICKERS)); rt = str(rng.choice(list(pd_by_rating), p=[.15, .35, .35, .15]))
        n = float(rng.uniform(20e6, 150e6)); p = pd_by_rating[rt]
        row(asset_type="Loan", entity=tk, sector=UNIVERSE[tk][1], notional=n, market_value=n * (1 - p * 0.45),
            rating=rt, pd=p, lgd=0.45)
    for _ in range(15):
        tk = str(rng.choice(TICKERS)); mv = float(rng.uniform(10e6, 80e6)); d = float(rng.uniform(2, 9))
        row(asset_type="Bond", entity=tk, sector=UNIVERSE[tk][1], notional=mv, market_value=mv,
            rating=str(rng.choice(["AA", "A", "BBB"])), duration=d, spread_duration=d * 0.9)
    for _ in range(10):
        tk = str(rng.choice(TICKERS)); sign = float(rng.choice([-1, 1]))
        row(asset_type="Swap", entity=tk, sector=UNIVERSE[tk][1], notional=100e6,
            market_value=float(rng.uniform(-2e6, 3e6)), dv01=sign * float(rng.uniform(10e3, 60e3)))
    for _ in range(8):
        tk = str(rng.choice(TICKERS)); und = float(rng.uniform(5e6, 40e6))
        row(asset_type="Option", entity=tk, sector=UNIVERSE[tk][1], notional=und, market_value=0.05 * und,
            delta=float(rng.choice([-0.5, -0.3, 0.3, 0.5])))
    for _ in range(7):
        tk = str(rng.choice(TICKERS)); mv = float(rng.uniform(5e6, 30e6))
        row(asset_type="Equity", entity=tk, sector=UNIVERSE[tk][1], notional=mv, market_value=mv)
    return pd.DataFrame(rows)
