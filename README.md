# S&P Global & CRISIL Campus Hackathon - Financial Risk Engine & Stress Testing

**Candidate Name:** Umang Aggarwal  
**College Email ID:** umang23567@iiitd.ac.in  
**College / Campus:** Indraprastha Institute of Information Technology Delhi (IIIT-Delhi)  
**Demo Video Link:** Add the final public/unlisted YouTube link before submission.
**Slide Deck:** Add `docs/presentation.pdf` (5–7 slides) before submission. The working draft is [docs/presentation_content.md](docs/presentation_content.md).

---

## 1. Project Overview / Problem Statement & Approach

Financial institutions operate under vast streams of unstructured text—news feeds, social media, and regulatory filings—containing real-time risk indicators long before they manifest in official quarterly statements. However, translating high-velocity textual noise into reliable, quantitative balance-sheet intelligence requires solving three critical challenges: **information deduplication**, **domain-specific financial sentiment analysis**, and **institutional risk-signal attribution**.

This project presents an explainable, end-to-end financial-risk prototype centered around a **Unified Shared AI/NLP Risk Engine** coupled with two downstream risk applications:
1. **Module A (Tactical Index Rebalancer):** A dynamic asset-allocation engine that continuously tilts S&P constituent weights based on real-time sentiment signals ($w_i' = w_i \times [1 + \alpha \cdot s_i]$) under portfolio turnover bounds.
2. **Module B (Strategic Wholesale Portfolio Stress-Testing):** A multi-asset stress-testing cockpit inspired by BCBS 239 data-lineage principles and Federal Reserve SR 11-7 model-governance practices. It monitors wholesale banking portfolios (Equities, Corporate Bonds, Loans with EAD/PD/LGD, and Swaps with DV01). High-impact event signals trigger illustrative macro/credit shocks, computing pre- and post-stress valuation waterfalls and loss contributions.

---

## 2. Architecture & Tech Stack

```
Data Ingestion (RSS Feeds, Replay Scenarios)
        │
        ▼
Cleaning & Deduplication (MD5 & Token Similarity)
        │
        ▼
Entity Resolution (Transformer NER & Security Master Fallback)
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
• Alpha-sentiment weight tilt    • Policy Gate (Impact >= 7.0, Conf >= 0.65)
• Bound-clamped turnover         • Multi-Factor Sensitivity Shocks
• Live weight tracking           • Illustrative Pre/Post Valuation Waterfall
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

1. **Security Master Dataset (`data/reference/security_master.json`):** Curated prototype universe of benchmark securities with ticker symbols, aliases, sectors, historical market betas, and credit ratings.
2. **Synthetic Wholesale Banking Portfolio (`data/portfolios/synthetic_portfolio.json`):** An approximately $366.6M wholesale banking book under the prototype valuation convention, featuring:
   - **Equities:** Market values and historical beta sensitivities.
   - **Corporate Bonds:** Market values, modified durations, credit ratings, and credit spread baselines.
   - **Wholesale Commercial Loans:** Exposure at Default (EAD), Probability of Default (PD), and Loss Given Default (LGD).
   - **Interest Rate Swaps:** Notional values and Dollar Value of a Basis Point (DV01) rate sensitivities.
3. **Deterministic Crisis & Operational Replay Feed (`data/demo/replay_feed.json`):** Curated scenarios spanning Boeing credit downgrades, Middle East maritime chokepoints, Fed emergency rate hikes, NVIDIA tech earnings, Citigroup regulatory fines, and non-material false-positive rumors.
4. **Financial PhraseBank (`data/external/financial_phrasebank/financial_phrasebank.csv`, local evaluation data):** A public labeled sentiment dataset used for a balanced 60-sentence FinBERT smoke evaluation. It is not required to run the demo; retain it in a submission only if its redistribution terms permit.
5. **Live RSS Feeds:** Ingestion connector for CNBC, Yahoo Finance, and MarketWatch financial news. The live feeds are external inputs and are not required for the deterministic demo.

### Data and prototype assumptions

- The security master, demo users, replay scenarios, and wholesale portfolio are curated or synthetic prototype inputs.
- Stress shocks and policy thresholds are illustrative assumptions, not calibrated production risk parameters.
- The replay feed contains multiple named sources for deterministic corroboration. RSS adapters provide an additional live source path when network access is available.
- Financial PhraseBank is used only for sentiment evaluation. The unlabeled news JSON collection is used only for ingestion/cleaning validation when a local subset is present.
- No proprietary, credentialed, or customer data is required to run the repository.

---

## 4. Quickstart & Installation

**Runtime:** Python 3.14 tested on macOS; Python 3.10+ is the intended compatible range.

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

### Optional Hugging Face Authentication

The application downloads the public FinBERT and event-classification models from Hugging Face. Anonymous access works, but Hugging Face may apply lower rate limits. To authenticate, create a Hugging Face token with **Read** permission at https://huggingface.co/settings/tokens.

For a one-time local setup on macOS or Linux, save the token in your shell profile:

```bash
echo 'export HF_TOKEN="your_read_token_here"' >> ~/.zshrc
source ~/.zshrc
```

If you use Bash instead of Zsh, replace `~/.zshrc` with `~/.bashrc`. On Windows PowerShell, save it as a persistent user environment variable:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "your_read_token_here", "User")
```

Restart the terminal after the PowerShell setup. After this one-time configuration, run the application normally:

```bash
python main.py
```

Never commit the token to the repository or place it directly in source code. `HF_HUB_OFFLINE=1` is optional and is not required for normal operation. The repository does not require a token for the public model downloads, subject to Hugging Face rate limits.

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

### Implementation Notes

- Raw input articles are persisted in SQLite before NLP processing. URL and content-hash uniqueness prevent the same article from being analyzed again after an application restart.
- Entity recognition uses an optional transformer NER model (`dslim/bert-base-NER`) to identify organization mentions, then links them to the curated security master for ticker, issuer, sector, and exposure metadata. Deterministic alias/ticker matching remains the fallback and is used by lightweight tests.
- The demo ingests multiple sources (RSS feeds from CNBC, Yahoo Finance, and MarketWatch plus curated replay sources) and clusters similar same-event reports. Formal stress execution requires at least two independent corroborating sources in addition to the impact and confidence thresholds.
- Module B's scenario selector only lists signals that already satisfy impact, confidence, and corroboration requirements, so the live demo does not present a scenario that will immediately be rejected.
- The dashboard and optional FastAPI service use FinBERT by default. If FinBERT cannot initialize or inference fails, the configured strict mode reports an error rather than silently producing a different sentiment result.
- Stress testing is policy-gated: signals below impact `7.0`, confidence `0.65`, or two independent corroborating sources are rejected and do not produce portfolio loss calculations.
- Module B only offers signals that satisfy the formal stress policy for execution. If no eligible signal is available, it displays an explanatory empty state instead of zero-valued charts. Accepted runs show loss attribution by asset class, issuer, sector, and position.
- All users share the global signal universe. Sarah and Marcus start with their watchlists as optional filters and can view all signals; portfolio stress access remains role-specific, while Admin has consolidated-portfolio and Securities Master access.
- Module A uses the logged-in user's watchlist as the allocation universe for Sarah and Marcus; Admin uses the full benchmark universe.
- Authentication is a synthetic demo login using the profiles in `data/demo/users_profiles.json`; it is not production identity management. Passwords should not be reused outside this demo.
- The automated suite is separated by component under `tests/` and includes focused checks for cleaning/persistence, entity resolution, event and impact scoring, corroboration, stress testing, and the compact six-example labeled fixture in `data/evaluation/evaluation_samples.json`. If the locally downloaded datasets are present under `data/external/`, `tests/test_external_datasets.py` evaluates a balanced 60-sentence Financial PhraseBank subset with FinBERT and runs 30 Reuters/CNBC/WSJ news JSON records through article cleaning. The PhraseBank result is a small smoke evaluation, not a statistically powered benchmark; the unlabeled news subset validates ingestion and cleaning only.

### Evaluation snapshot

The current local evaluation reports:

- Financial PhraseBank: 60 balanced sentences, FinBERT accuracy **81.7%**.
- Component and integration suite: **13 passed, 1 skipped** in the last full run. The skipped check requires the optional unlabeled US-news JSON subset.

These are prototype smoke-evaluation results, not claims of production model performance.

### Submission checklist

Submit one public GitHub repository containing:

1. Complete source code under `src/` and the entrypoint `main.py`.
2. `requirements.txt`, `LICENSE`, and this README with reproducible setup/run commands.
3. The demo/reference/evaluation data under `data/`, including synthetic portfolio inputs and the selected evaluation data permitted for redistribution.
4. A 5–7 slide PDF or PPTX at `docs/presentation.pdf` covering the problem, architecture, implementation, results, impact, and limitations.
5. A public or unlisted YouTube demo link added above. The walkthrough should stay within the submission guidance time limit and show ingestion, NLP signals, and at least one downstream module.

Before submitting, verify that the GitHub repository is public, the demo and deck links work in an incognito browser, no API tokens or credentials are committed, and the exact Quickstart command works from a clean environment.

### Known limitations and next steps

- The portfolios, security master, replay scenarios, and stress assumptions are synthetic or curated for demonstration.
- Event classification and impact scoring remain prototype rule/model hybrids and should be calibrated and reviewed before production use.
- The SQLite store and Streamlit authentication are suitable for a demo, not production identity, concurrency, or deployment.
- A larger held-out sentiment/event/entity benchmark and historical market-outcome validation would be needed for production model claims.

---

## 5. Key Results & Domain Impact

- **End-to-End Operational Pipeline:** Ingests articles, removes duplicates, resolves corporate entities, scores FinBERT sentiment, classifies events into 8 taxonomies, and computes an explainable 1-10 impact score through a deterministic prototype workflow.
- **Model Governance Prototype:** Features an automated policy gate inspired by SR 11-7 practices that filters low-materiality noise (such as cosmetic product rumors), while escalating genuine systemic threats (e.g., Fitch rating cuts) to stress testing. This is not a claim of regulatory compliance.
- **Multi-Asset Stress Loss Attribution:** In the simulated Boeing credit downgrade scenario, the engine accurately isolates single-name credit spread widening (+220 bps) and loan PD jumps (2.5x), illustrating a $13.1M (-3.59%) loss across loan provisions and bond prices while sparing uncorrelated market positions.
- **Tactical Dynamic Rebalancing:** Successfully redistributes portfolio weights according to sentiment velocity while maintaining strict turnover bounds.
