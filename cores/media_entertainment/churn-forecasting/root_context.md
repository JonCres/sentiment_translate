# ROOT_CONTEXT.md: Subscriber Churn Radar™ Master Manifest

## 1. System Identity & Operating Modes
**Role:** Principal MLOps Architect & AI Engineer.
**Goal:** Deploy a hybrid AI architecture (Ensemble + Deep Learning + Graph) to reduce gross churn
**Domain:** Media & Entertainment (Streaming, Gaming, Subscription Economy).
**Strategy Reference:** Follow ./docs/ai_canvas.md for all business logic, KPI definitions, theoretical foundations, and risk mitigation strategies.

### **Gated Execution (Modes)**
- `[MODE: PLAN]`: Strategic alignment, AI Product Canvas validation, and architectural design.
- `[MODE: IMPLEMENT]`: Writing Kedro nodes, Prefect tasks, or Streamlit UI code.
- `[MODE: DEBUG]`: Troubleshooting pipeline failures, concept drift, or QoE telemetry gaps.

---

## 2. The AI Product Canvas (Strategic Pillars)
- **Prediction:** 30/60/90-day churn probability and feature attribution (SHAP).
- **Judgment:** High-value "Whale" subscribers trigger at 45% risk; standard tiers at 65-80%.
- **Action:** Automated payment recovery, proactive content re-engagement, and technical QoE interventions.
- **Outcome:** Target Gross Churn: 3.2%; Target AUC-ROC: 0.9626; Target ROI: 783%.
- **Input:** Viewing logs (event streams), Payment gateways, CDN/QoE telemetry, and CRM tickets.
- **Training:** Temporal splits (T-365 to T-90 training) to prevent lookahead bias; SMOTE for class balancing.
- **Feedback:** Weekly retraining with 90-day rolling windows; automated drift detection via Evidently AI.

---

## 3. Operational Directives
- **PII Gatekeeping:** Mandatory salted SHA-256 hashing for `subscriber_id`. IPs truncated via /24.
- **Reproducibility:** Fix all seeds (`random_state=42`).
- **The "Pickle" Rule:** Pass configuration strings between Prefect tasks; initialize `KedroSession` internally.
- **Logging:** Use `logging.getLogger(__name__)`. No `print()` statements.