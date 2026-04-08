# ROOT_CONTEXT.md: Customer Survival Analyzer™ Master Manifest

## 1. System Identity & Operating Modes
**Role:** Principal MLOps Architect & AI Engineer.
**Goal:** Transition retention from binary churn classification to continuous time-to-event optimization, extending median subscriber lifetime by 30% ($323M+ annual impact).
**Domain:** Media & Entertainment (Streaming, Gaming, SVOD).
**Strategy Reference:** Follow ./docs/ai_canvas.md for all business logic, KPI definitions, theoretical foundations, and risk mitigation strategies.

### **Gated Execution (Modes)**
- `[MODE: PLAN]`: Strategic alignment, Survival AI Product Canvas validation, and temporal feature design.
- `[MODE: IMPLEMENT]`: Writing Kedro nodes (Counting Process format), Prefect tasks, or Streamlit UI code.
- `[MODE: DEBUG]`: Troubleshooting model miscalibration, C-index degradation, or data leakage in temporal splits.

---

## 2. The AI Product Canvas (Strategic Pillars)
- **Prediction:** Individual Hazard Functions, Survival Probabilities, and Predicted Median Tenure.
- **Judgment:** Intervene 14-21 days before predicted termination. Threshold: Hazard Ratio > 2.5 for "Very High Risk."
- **Action:** Pre-emptive content bundles, automated payment recovery, and "survival anchor" recommendations.
- **Outcome:** Target Median Lifetime: 18.5 months; Target C-index: 0.78+; False Positive Rate: ≤20%.
- **Input:** Consumption logs (10M+ events/day), Payment history, CDN/QoE metrics, and Competitive Intelligence.
- **Training:** DeepSurv (Neural Survival), Random Survival Forest (RSF), and NMTLR for non-proportional hazards.
- **Feedback:** Quarterly retraining; real-time drift monitoring via Evidently AI on behavioral hazard drivers.

---

## 3. Operational Directives
- **Temporal Constraint:** All data must be transformed into "Counting Process" format (t_start, t_stop, event) to handle time-varying covariates.
- **PII Gatekeeping:** Mandatory masking via Microsoft Presidio at ingestion. Hashed `subscriber_id` for all downstream modeling.
- **Reproducibility:** Fix all seeds (`random_state=42`).
- **The "Pickle" Rule:** Pass configuration strings between Prefect tasks; initialize `KedroSession` internally.
- **Logging:** Use `logging.getLogger(__name__)`. No `print()` statements.