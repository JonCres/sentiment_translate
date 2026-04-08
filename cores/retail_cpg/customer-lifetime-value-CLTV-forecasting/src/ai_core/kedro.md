# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Bifurcates logic into `transactional_stream` (BG/NBD) and `subscription_stream` (Weibull AFT).
- `feature_engineering`: 
    - **RFM Path:** Recency, Frequency (repeat-only), and Monetary (repeat-only) vectors.
    - **Engagement Path:** Session decay rates and content entropy (LSTM sequences).
- `data_science`: Ensembles LightGBM with Monte Carlo Dropout for uncertainty quantification.
- `explainability_insights`: (See Section 2).

## 2. Explainability & SLM Integration
- **Global Explainability:** Compute Gini coefficients and SHAP values to rank-order value drivers (e.g., "Discount Depth" vs "Category Affinity").
- **Local Explainability:** SHAP waterfall plots for individual customer value breakdown.
- **SLM Narrative Engine:** Use a **Small Language Model (SLM)** (e.g., Phi-3) to interpret high-value residuals.
    - **Insight Translation:** Convert SHAP vectors into business narratives: *"Customer C-45821 is a 'Rising Star'; despite low frequency, their high category diversity and low discount sensitivity suggest a 45% probability of migrating to Q1."*

## 3. Coding Standards
- **Validation:** Use **Pandera** to enforce that `net_transaction_value` accounts for returns and `p_alive` stays in [0, 1].
- **Purity:** Nodes must be pure functions. Exclude first purchase from monetary calculation to avoid CAC distortion.