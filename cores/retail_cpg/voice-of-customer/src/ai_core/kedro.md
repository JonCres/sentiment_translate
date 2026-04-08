# Pipeline Layer (Kedro)

**Role:** Senior Data Scientist & Data Engineer.

## 1. Modular Architecture
- `data_processing`: Standardization of multi-channel inputs into a "Unified Interaction" skeleton.
- `text_intelligence`: Transformer-based ABSA (SetFit/BERT) with aspect-context tokenization.
- `multimodal_emotion`: 1D-CNN (Audio) + 3D-CNN (Video) + Cross-attention transformers for MER alignment.
- `behavioral_context`: Session-based transformers for implicit sentiment (rage clicks, abandonment).
- `graph_analytics`: GNNs for root cause propagation across the Customer Experience Graph.

## 2. Explainability & SLM Integration
- **Model Explainability:** Use **SHAP/LIME** to identify which words or acoustic features (pitch/jitter) triggered a "Frustration" label.
- **SLM Narrative Engine:** Use a **Small Language Model (SLM)** (e.g., Phi-3 or Mistral) to interpret complex aspect-sentiment clusters.
    - **Insight Translation:** Convert raw vectors into business logic: *"SKU-123 is seeing a 40% spike in 'zipper failure' mentions in the Northeast region; likely a localized logistics/batch defect."*

## 3. Coding Standards
- **Validation:** Use **Pandera** to enforce interaction schemas.
- **Purity:** Nodes must be pure functions. Enforce 90-day observation windows to prevent look-ahead bias.
- **Vectorization:** Use NumPy/Pandas/Polars; strictly avoid `for` loops on interaction batches.