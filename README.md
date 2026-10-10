# S&P Global & CRISIL Campus Hackathon - Financial Risk Engine & Stress Testing

**Candidate Name:** Umang Aggarwal  
**College Email ID:** umang23567@iiitd.ac.in  
**College / Campus:** Indraprastha Institute of Information Technology Delhi (IIIT-Delhi)  
**Demo Video Link:** [YouTube / Unlisted Link Placeholder]  
**Slide Deck Link:** [docs/presentation_content.md](file:///Users/umangaggarwal/Desktop/S&P%20x%20Crisil%20Hackathon%20copy/docs/presentation_content.md)

---

## 1. Project Overview / Problem Statement & Approach

Financial institutions operate under vast streams of unstructured text—news feeds, social media, and regulatory filings—containing real-time risk indicators long before they manifest in official quarterly statements. However, translating high-velocity textual noise into reliable, quantitative balance-sheet intelligence requires solving three critical challenges: **information deduplication**, **domain-specific financial sentiment analysis**, and **institutional risk-signal attribution**.

This project presents an enterprise-grade, end-to-end platform centered around a **Unified Shared AI/NLP Risk Engine** coupled with two distinct downstream risk applications:
1. **Module A (Tactical Index Rebalancer):** A dynamic asset-allocation engine that continuously tilts S&P constituent weights based on real-time sentiment signals ($w_i' = w_i \times [1 + \alpha \cdot s_i]$) under portfolio turnover bounds.
2. **Module B (Strategic Wholesale Portfolio Stress-Testing):** A Basel BCBS 239 & Federal Reserve SR 11-7 compliant multi-asset stress-testing cockpit that monitors wholesale banking portfolios (Equities, Corporate Bonds, Loans with EAD/PD/LGD, and Swaps with DV01). High-impact event signals trigger multi-factor macro/credit shocks, computing pre- and post-stress valuation waterfalls and loss contributions.

---

## 2. Architecture & Tech Stack

```
Data Ingestion (RSS Feeds, Replay Scenarios)
        │
        ▼
Cleaning & Deduplication (MD5 & Token Similarity)
        │
        ▼
Entity Resolution (Security Master & spaCy/Regex Rules)
        │
        ▼
Shared NLP Risk Engine
├── Financial Sentiment: ProsusAI/finbert & Loughran-McDonald Lexicon
├── Event Classification: 8-Class Financial Taxonomy
└── Impact Severity: 1-10 Explainable Risk Score
        │
        ▼
Global Risk Signal DB (SQLite & FastAPI Service)
        │
        ├────────────────────────────────┐
        ▼                                ▼
Module A: Index Rebalancer       Module B: Wholesale Stress Testing
• Alpha-sentiment weight tilt    • Policy Gate (Impact >= 7.0, Conf >= 0.70)
• Bound-clamped turnover         • Multi-Factor Sensitivity Shocks
• Live weight tracking           • Basel Pre/Post Valuation Waterfall
        │                                │
        └────────────────┬───────────────┘
                         ▼
        Interactive Dashboard (Streamlit & Plotly)
```

### Key Technologies:
- **NLP & AI Models:** PyTorch, Hugging Face Transformers (`ProsusAI/finbert`), Loughran-McDonald Financial Lexicon.
- **Backend & APIs:** FastAPI, Pydantic v2, SQLite3, Uvicorn.
- **Financial Math & Processing:** Pandas, NumPy, Scikit-Learn.
- **Interactive UI:** Streamlit, Plotly Interactive Charts.
- **Testing:** Pytest automated test suite.

---

## 3. Dataset Used

1. **Security Master Dataset (`data/security_master.json`):** Curated universe of 15 institutional benchmark equities (Apple, Microsoft, NVIDIA, JPMorgan, Boeing, Citigroup, ExxonMobil, etc.) with ticker symbols, aliases, sectors, historical market betas, and credit ratings.
2. **Synthetic Wholesale Banking Portfolio (`data/synthetic_portfolio.json`):** A $365M wholesale banking book featuring:
   - **Equities:** Market values and historical beta sensitivities.
   - **Corporate Bonds:** Market values, modified durations, credit ratings, and credit spread baselines.
   - **Wholesale Commercial Loans:** Exposure at Default (EAD), Probability of Default (PD), and Loss Given Default (LGD).
   - **Interest Rate Swaps:** Notional values and Dollar Value of a Basis Point (DV01) rate sensitivities.
3. **Deterministic Crisis & Operational Replay Feed (`data/replay_feed.json`):** Curated scenarios spanning Boeing credit downgrades, Middle East maritime chokepoints, Fed emergency rate hikes, NVIDIA tech earnings, Citigroup regulatory fines, and non-material false-positive rumors.
4. **Live RSS Feeds:** Ingestion connector for CNBC, Yahoo Finance, and MarketWatch financial news.

---

## 4. Quickstart & Installation

**Runtime:** Python 3.10+ / 3.11 / 3.12 / 3.14 on macOS / Linux / Windows

### Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd IIIT-Delhi-Umang-Aggarwal-Hackathon

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
You can run the entire platform using the unified entrypoint:
```bash
python main.py
```
Or launch the Streamlit Dashboard directly:
```bash
streamlit run src/dashboard/app.py
```

### Run the Standalone FastAPI Backend (Optional)
```bash
uvicorn src.api.server:app --reload --port 8000
```

### Run Automated Tests
```bash
PYTHONPATH=. pytest tests/
```

---

## 5. Key Results & Domain Impact

- **End-to-End Operational Pipeline:** Ingests articles, removes duplicates, resolves corporate entities, scores FinBERT sentiment, classifies events into 8 taxonomies, and computes an explainable 1-10 impact score in single milliseconds per story.
- **Institutional Risk Governance (SR 11-7):** Features an automated policy gate that filters low-materiality noise (such as cosmetic product rumors), while escalating genuine systemic threats (e.g., Fitch rating cuts) to stress testing.
- **Multi-Asset Stress Loss Attribution:** In the simulated Boeing credit downgrade scenario, the engine accurately isolates single-name credit spread widening (+220 bps) and loan PD jumps (2.5x), illustrating a $13.1M (-3.59%) loss across loan provisions and bond prices while sparing uncorrelated market positions.
- **Tactical Dynamic Rebalancing:** Successfully redistributes portfolio weights according to sentiment velocity while maintaining strict turnover bounds.
