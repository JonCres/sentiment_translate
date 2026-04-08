# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Bifurcates logic into `contractual_stream` (SVOD) and `non_contractual_stream` (TVOD/Gaming).
- `feature_engineering`: 
    - **RFM Path:** Recency, Frequency, and Monetary vectors.
    - **Engagement Path:** Binge velocity, content exhaustion signals, and habit stability.
- `data_science`: Trains probabilistic baselines (BG/NBD) and fits XGBoost on residuals for refinement.
- `explainability_insights`: (See Section 2).

## 2. Explainability & SLM Integration
- **Model Explainability:** Use **SHAP** to identify drivers of "Silent Attrition" (e.g., Engagement Recency vs. Billing Status).
- **SLM Narrative Engine:** Integrate a **Small Language Model (SLM)** (e.g., Phi-3) to interpret complex behavioral archetypes.
    - **Insight Generation:** Convert SHAP vectors into business narratives: *"Customer C-88421 is a 'Binge-and-Burnout' risk. Despite a successful renewal, their engagement velocity has dropped 80% since completing 'Season 4'. The SLM recommends a 'Bridge Content' offer to preserve $387 in 12-month value."*

## 3. Coding Standards
- **Validation:** Use **Pandera** to enforce that `transaction_date` is sequential and `p_alive` remains in [0, 1].
- **Purity:** Nodes must be pure functions. Enforce strict temporal splits (Training: T-24 to T-6 months) to prevent look-ahead bias.