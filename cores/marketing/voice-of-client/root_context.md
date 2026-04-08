# AI CORE MASTER MANIFEST: VOICE OF THE CLIENT

## 1. Strategic Mandate
This project implements the **Voice of the Client** AI Core, a B2B Strategic Account Intelligence engine.
**CRITICAL INSTRUCTION:** All architectural decisions, model selections, and business logic must strictly adhere to the definitions found in `docs/ai_product_canvas.md`.

**Primary Objective:** Transform unstructured B2B feedback (NPS, Support, CRM) into predictive churn risk intelligence and actionable revenue-protection strategies.

## 2. Intelligence Architecture
The system operates on a **Hybrid NLP + Tabular** architecture:
1.  **Ingestion:** Multi-channel data (Survey, CRM, Support) keyed by `account_id`.
2.  **Sanitization:** Mandatory PII redaction (Presidio) before persistence.
3.  **NLP Core:**
    *   **Taxonomy:** Zero-Shot Classification (DeBERTa).
    *   **Sentiment:** Aspect-Based Sentiment Analysis (PyABSA).
    *   **Themes:** Dynamic Topic Modeling (BERTopic).
4.  **Predictive Core:**
    *   **Churn:** Gradient Boosting (XGBoost/LightGBM) with SMOTE for class imbalance.
    *   **Explainability:** SHAP (Shapley Additive Explanations) for all predictions.
5.  **Synthesis:** Small Language Models (SLMs) to convert SHAP/Sentiment data into human-readable narratives.

## 3. Technology Stack
*   **Pipeline Framework:** Kedro (Data engineering & model pipelines).
*   **Orchestration:** Prefect (Workflow automation & monitoring).
*   **Application:** Streamlit (Visualization & Action interface).
*   **ML Libraries:** HuggingFace Transformers, PyABSA, XGBoost, SHAP, Presidio.
*   **Monitoring:** Evidently AI (Drift detection).

## 4. Gated Execution Modes
To ensure data safety and compliance, the system operates in two distinct modes:

### Mode A: Development (Strict)
*   **Data:** Synthetic B2B datasets or anonymized snapshots only.
*   **PII:** Presidio pipeline **MUST** run on all text inputs.
*   **Outputs:** Local artifacts, mock API responses.

### Mode B: Production (Operational)
*   **Data:** Live connection to CRM/Data Lake (Snowflake/S3).
*   **PII:** Real-time redaction before Feature Store ingestion.
*   **Outputs:** Writes to CRM fields, Slack Alerts, Production Dashboard.

## 5. Global Development Rules
1.  **Account-Centricity:** All analysis must aggregate to the `account_id`. Individual `user_id` sentiment is secondary to the "Decision Maker" weight.
2.  **Revenue-Weighting:** All risk scoring must factor in `contract_value` (ARR).
3.  **Explainability First:** No "Black Box" predictions. Every churn score must be accompanied by the top 3 driving features.