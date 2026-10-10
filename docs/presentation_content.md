# S&P Global & CRISIL Campus Hackathon 2026
## Presentation Slide Deck (7 Slides)
**Project Title:** AI/NLP Financial Risk Engine & Stress Testing Platform  
**Candidate Name:** Umang Aggarwal  
**Institution:** IIIT-Delhi  

---

### Slide 1: Title & Overview
- **Header:** Institutional AI/NLP Financial Risk Engine & Event-Driven Stress Testing Platform
- **Sub-header:** Transforming High-Velocity Financial Unstructured Feeds into Basel-Compliant Portfolio Intelligence
- **Candidate:** Umang Aggarwal | IIIT-Delhi
- **Key Modules Solved:** 
  - Central Shared AI/NLP Risk Engine (FinBERT + 8 Event Taxonomies + 1-10 Explainable Impact)
  - Module A: Tactical High-Frequency Index Rebalancer
  - Module B: Strategic Multi-Asset Wholesale Banking Portfolio Stress-Tester

---

### Slide 2: Problem & Approach
- **Business Challenge:** 
  - Over 80% of institutional risk signals appear in unstructured text (news, social, regulatory actions) hours before appearing in pricing or filings.
  - Naive systems either hallucinate false alarms or suffer extreme latency by recalculating NLP models per-user.
- **Our Approach:**
  - **Shared Central Processing:** Ingest and parse each story once into a canonical risk ledger.
  - **Multi-Factor Explainable Impact:** Decouple simple sentiment from systemic materiality (do not equate negative sentiment with high market impact).
  - **Downstream Decoupling:** Both tactical rebalancing (Module A) and capital stress-testing (Module B) consume clean, versioned signals asynchronously.

---

### Slide 3: System Architecture & Data Flow
- **End-to-End Pipeline:**
  1. *Ingestion:* Real-time RSS feeds (CNBC, Yahoo) + Deterministic Crisis Replay.
  2. *Normalization & Dedupe:* Content hashing (MD5) + token-based near-duplicate clustering.
  3. *Entity Resolution:* Security Master mapping (15 core tickers, aliases, betas, ratings).
  4. *Shared NLP Engine:* FinBERT ($s = P(\text{pos}) - P(\text{neg})$) + 8-class taxonomy + evidence span extraction.
  5. *Persistence & API:* SQLite immutable store + FastAPI endpoints.
  6. *Downstream Applications:* Tactical Index Rebalancer + Basel Stress-Testing Cockpit.

---

### Slide 4: Implementation Highlights
- **FinBERT Sentiment Inference:** Evaluates probabilities across positive, neutral, and negative vectors to capture market nuance.
- **Explainable Impact Scoring (1–10):** Transparent multi-factor engine incorporating:
  $$\text{Impact} = (\text{Base Materiality} + \text{Sentiment Modifier} + \text{Entity Directness}) \times \text{Source Credibility}$$
- **Wholesale Multi-Asset Valuation Mathematics:**
  - *Equities:* $\Delta P = \text{Market Value} \times \beta \times \Delta R_{\text{market}}$
  - *Bonds:* $\Delta P \approx -\text{Duration} \times (\Delta y + \Delta s_{\text{credit}})$
  - *Wholesale Loans:* $\Delta \text{Expected Loss} = \text{EAD} \times (\text{PD}_{\text{stressed}} \times \text{LGD}_{\text{stressed}} - \text{PD}_0 \times \text{LGD}_0)$
  - *Interest Rate Swaps:* $\Delta P = -\text{DV01} \times \Delta r$

---

### Slide 5: Key Results & Stress Testing Walkthrough
- **Scenario Tested:** Fitch Rating Downgrade of Boeing (BA) senior debt to BBB-.
  - *NLP Signal:* Impact: 9.9/10, FinBERT Sentiment: -0.82, Event: Credit Event.
  - *Policy Gate:* **ACCEPTED** (Trigger threshold $\ge 7.0$).
  - *Wholesale Impact:* Total Portfolio Loss of **-$13.15M (-3.59%)** on a $365M book.
  - *Attribution:* Loan provisions jumped on 2.5x PD stress; Boeing 5-year bonds dropped on +220 bps spread widening; unaffected market sectors remained insulated.
- **Module A Rebalancing Result:** Successfully adjusted 12 S&P constituents according to sentiment velocity under turnover bounds.

---

### Slide 6: Domain Impact & Risk Governance (SR 11-7 / BCBS 239)
- **Model Risk Management (SR 11-7):**
  - Incorporated an explicit **Policy Gate** that rejected cosmetic rumors (e.g., Apple pastel watch straps, impact 2.6) with logged audit reasons, preventing spurious stress runs.
- **BCBS 239 Data Aggregation:**
  - Immutable signal versioning, model tracking, and complete traceable evidence text spans from raw article to stressed balance-sheet P&L.
- **Enterprise Efficiency:** Single shared inference cycle eliminates redundant GPU computation across thousands of institutional desks.

---

### Slide 7: Limitations & Future Roadmap
- **Current Prototype Scope:** 15 benchmark companies in Security Master, simplified one-way duration approximations for corporate debt.
- **Future Enhancements:**
  - Integration of high-frequency order book microstructure data (L2/L3 quotes).
  - Transition from SQLite to distributed Kafka + PostgreSQL JSONB streaming architecture.
  - Machine learning based calibration of macroeconomic shock transmission matrices.
