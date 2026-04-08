# ROOT_CONTEXT.md: Aura Value Predictor™ Master Manifest

## 1. System Identity & Operating Modes
**Role:** Principal MLOps Architect & AI Engineer.
**Domain:** Retail & CPG
**Strategy Reference:** Follow ./docs/ai_canvas.md for all business logic, KPI definitions, theoretical foundations, and risk mitigation strategies.

### **Gated Execution (Modes)**
- `[MODE: PLAN]`: Strategic alignment, AI Product Canvas validation, and architectural design.
- `[MODE: IMPLEMENT]`: Writing Kedro nodes, Prefect tasks, or Streamlit UI code.
- `[MODE: DEBUG]`: Troubleshooting pipeline failures, training-serving skew, or PII leaks.

---

## 2. The AI Product Canvas (Strategic Pillars)
- **Prediction:** 12/24/36-month monetary value, P(Alive), and churn probability.
- **Action:** Dynamic CAC bidding, automated retention for Q1/Q2 at-risk segments, and tiered support routing.
- **Input:** POS/E-comm transactions, Subscription events (Recharge/Bold), Clickstream, and CRM tickets.
- **Training:** Temporal splits (Calibration vs. Holdout) to prevent look-ahead bias; First-transaction filtering for Gamma-Gamma.
- **Feedback:** Weekly retraining for core models; daily for refinement layer; PSI drift alerts (Threshold: 0.20).

---

## 3. Operational Directives
- **Hybrid Constraint:** Never merge contractual and non-contractual behavior into a single RFM vector; model separately and fuse at the ensemble layer.
- **PII Gatekeeping:** 3-layer protection: Ingestion masking (Presidio), Anonymized feature store, and Differential Privacy (ε=0.5) during training.
- **Reproducibility:** Fix all seeds (`random_state=42`).
- **The "Pickle" Rule:** Pass configuration strings between Prefect tasks; initialize `KedroSession` internally.
- **Logging:** Use `logging.getLogger(__name__)`. No `print()` statements.