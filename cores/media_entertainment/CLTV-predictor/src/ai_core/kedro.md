# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Sessionization of raw SDK heartbeats (30-min gap heuristic) and cross-platform identity resolution.
- `feature_engineering`: 
    - **Sequential Path:** Fixed-length tensors for LSTM (Attention sequences).
    - **Relational Path:** ALS-based Content Embeddings (50-dimension vectors).
    - **Monetary Path:** RFM-T aggregates (Recency, Frequency, Monetary, Tenure).
- `data_science`: Ensemble of XGBoost (SVOD), LSTM (AVOD), and Factorization Machines (TVOD) fused via Tweedie meta-learner.
- `explainability_insights`: (See Section 2).

## 2. Explainability & SLM Integration
- **Global Explainability:** Compute SHAP values to identify cross-platform multipliers and device-type influence (e.g., CTV vs. Mobile).
- **Local Explainability:** LIME for individual "Whale" subscriber value attribution.
- **SLM Insight Engine:** Integrate a **Small Language Model (SLM)** (e.g., Phi-3 or Mistral) to interpret the "Content-Value Correlation."
    - **Insight Generation:** Convert SHAP vectors and content affinity scores into natural language narratives for Content Acquisition: *"Genre 'Prestige Drama' is driving a 2.3x CLTV multiplier due to high rewatch velocity and CTV completion rates, justifying a 15% budget increase."*

## 3. Coding Standards
- **Validation:** Use **Pandera** to enforce that `ad_completion_rate` is in [0, 1] and `billing_cycle_revenue` is non-negative.
- **Purity:** Nodes must be pure functions. Enforce temporal splitting to avoid data leakage from future tentpole events.