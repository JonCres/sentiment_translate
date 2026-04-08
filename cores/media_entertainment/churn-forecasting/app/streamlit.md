# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **Subscriber Search:** Input hashed ID to view 30/60/90-day risk trajectory.
- **Explainability Panel:** 
    - Interactive SHAP waterfall plots for local risk attribution.
    - **SLM Insight Box:** Displays the SLM-generated "Subscriber Narrative" for quick human triage.
- **Intervention Queue:** Real-time list of "Critical" tier subscribers with recommended actions (e.g., "Send 30% Discount").
- **QoE Heatmap:** Visualization of CDN performance vs. churn spikes by region.

## 2. Performance & Caching
- **@st.cache_resource**: Load the CNN-BiLSTM and XGBoost models into memory.
- **@st.cache_data**: Cache aggregated cohort analytics from the Kedro catalog.

## 3. UX Standards
- **Color Coding:** Critical (Red), High (Orange), Medium (Yellow), Low (Green).
- **Interactivity:** Allow Marketing Ops to "Override" an intervention and feed that label back into the retraining loop.