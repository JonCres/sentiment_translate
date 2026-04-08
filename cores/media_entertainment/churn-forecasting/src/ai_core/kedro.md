# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Cleans viewing logs and standardizes Activity Schema.
- `feature_engineering`: 
    - **Ensemble Path:** Sliding window aggregates (RFM, velocity metrics).
    - **Deep Learning Path:** 3D Tensor construction (Samples × Time Steps × Features) for BiLSTM.
    - **Graph Path:** Synthetic similarity graphs based on co-consumption.
- `data_science`: Hybrid soft-voting ensemble (XGBoost/LightGBM) + CNN-BiLSTM + GAT.
- `explainability_insights`: (See Section 2).
- `monitoring`: data drift and model outdated detection

## 2. Explainability & SLM Integration
- **Global Explainability:** Compute SHAP values for the ensemble layer to identify macro-churn drivers (e.g., "Buffering Ratio").
- **Local Explainability:** LIME/SHAP for individual subscriber risk profiles.
- **SLM Narrative Engine:** Use a **Small Language Model (SLM)** (e.g., Phi-3 or Mistral-7B) to ingest SHAP values and behavioral logs.
    - **Output:** Generate natural language "Subscriber Narratives" for Marketing Ops (e.g., *"Subscriber X is at critical risk due to a 60% drop in drama consumption and 3 recent playback errors."*).

## 3. Coding Standards
- **Validation:** Use **Pandera** decorators to ensure `rebuffer_ratio` is within [0, 1] and `session_duration` is non-negative.
- **Purity:** Nodes must be pure functions. No global state.
