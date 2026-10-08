# Build plan (Oct 5 night -> Oct 11 deadline)

Scaffold already runs end to end. Remaining time goes to real data, credibility, and polish.

| Day | Goal | Done when |
|---|---|---|
| Oct 6 | **Real data.** Kaggle financial news + tweets CSV into `data/`; yfinance prices (`fetch_yfinance`); keep a small synthetic set for the demo | `python main.py all` runs on real text and prices |
| Oct 7 | **Fix NLP on real text.** Hand-label ~150 headlines (event + sentiment); evaluate vs naive; tune lexicon or enable FinBERT (`USE_FINBERT=1`) | honest accuracy/F1 table from real labels |
| Oct 8 | **Contagion + validation.** Calibrate graph weights (e.g. return correlation); real rank-IC; ablation: naive vs engine vs graph | validation numbers you can defend |
| Oct 9 | **Dashboard polish + stress UX.** Event picker, before/after waterfall, rumour-blocked panel; screenshots | clean 5-min demo path |
| Oct 10 | **Deck (5-7 slides), architecture check, record demo (<5 min), README numbers, test links in incognito, unlisted YouTube** | all assets linked |
| Oct 11 | Buffer: fresh-clone test on a clean venv, set repo public, submit before deadline | submitted early |

Commit incrementally every day (guidelines ask for visible history).

## Jury Q&A prep
- Why a lexicon, not FinBERT? Explainable, offline, instant; FinBERT is a switch.
- Is the graph real? Illustrative; calibrate and state it.
- Why block rumours? Credibility x corroboration before expensive actions.
- What would you do with more time? Learned edge weights, entity linking, intraday data.
