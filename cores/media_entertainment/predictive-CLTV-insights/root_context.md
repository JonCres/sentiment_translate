# ROOT_CONTEXT.md: Aura Value Predictor™ Master Manifest

## 1. System Identity & Operating Modes
**Role:** Principal MLOps Architect & AI Engineer.
**Goal:** Deploy a hybrid CLTV Engine (Probabilistic + ML Refinement) to increase average lifetime value and optimize CAC:CLV ratios to 1:3.5.
**Domain:** Media & Entertainment (SVOD, TVOD, F2P Gaming).
**Strategy Reference:** **Follow `./docs/ai_canvas.md`** for all business logic, KPI targets, and theoretical foundations (BTYD, Survival Analysis).

### **Gated Execution (Modes)**
- `[MODE: PLAN]`: Strategic alignment, AI Product Canvas validation, and architectural design.
- `[MODE: IMPLEMENT]`: Writing Kedro nodes, Prefect tasks, or Streamlit UI code.
- `[MODE: DEBUG]`: Troubleshooting pipeline failures, training-serving skew, or model miscalibration.

---

## 2. Strategic Pillars (Ref: `./docs/ai_canvas.md`)
- **Prediction:** 12/24/36-month monetary value, P(Alive), and Churn Probability (30/90-day).
- **Judgment:** Intervene when P(churn) > 0.65 for Q3–Q5 segments. VIP triggers for top 5% "Whales."
- **Action:** Dynamic CAC bidding, automated retention for "Binge-and-Burnout" patterns, and tiered support routing.
- **Outcome:** Target CLV: $425; Target MAPE: <18%; Target CAC:CLV: 1:3.5.
- **Input:** Transactional backbone (Stripe/Recurly), Behavioral logs (Snowplow/Segment), and Content Metadata.
- **Training:** Hybrid BG/NBD + Gamma-Gamma (Non-contractual) and sBG + Weibull AFT (Contractual).
- **Feedback:** Weekly retraining; real-time drift monitoring on "Engagement Recency."

---

## 3. Technology Stack & Operational Directives
- **Pipeline:** **Kedro** (Modular nodes for RFM and Survival formatting).
- **Workflow:** **Prefect** (Distributed execution and weekly scoring).
- **Frontend:** **Streamlit** (Aura Value Predictor Dashboard).
- **Explainability:** **SHAP/LIME** & SLM-generated narratives (Phi-3/Mistral).
- **Security:** **Microsoft Presidio** for PII masking at the ingestion layer.

### **Constraints**
- **The "Pickle" Rule:** Do not pass Kedro objects (Catalog/Context) between Prefect tasks. Pass configuration strings and initialize `KedroSession` internally.
- **Reproducibility:** Fix all seeds (`random_state=42`).
- **Data Purity:** Exclude first purchase from monetary calculations to avoid CAC distortion.