# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **Value Trajectory Explorer:** Search by `customer_id` to view 12/24/36-month CLV curves with 95% confidence intervals.
- **Segment Migration (Sankey):** Visualize customer movement between CLV Quintiles (Q1–Q5) over time.
- **Explainability & SLM Box:** 
    - Display SHAP waterfall plots for "Whale" attribution.
    - **SLM Narrative:** A text area showing the SLM-generated "Retention Strategy" based on the primary risk factor (e.g., Technical Churn vs. Content Exhaustion).
- **CAC:CLV ROI:** Real-time channel efficiency scorecard comparing acquisition cost to predicted lifetime value.

## 2. Performance & Caching
- **@st.cache_resource**: Load the serialized XGBoost refinement and probabilistic core models.
- **@st.cache_data**: Cache aggregated cohort metrics and "Gold Layer" feature tables from Delta Lake.

## 3. UX Standards
- **Interactivity:** Filter by "Acquisition Channel," "Behavioral Archetype," and "Geography."
- **Confidence Gating:** Gray out or flag predictions where the 95% CI width exceeds 75% of the point estimate (low-confidence targeting).