# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **CLV Explorer:** Search by `customer_id` to view 12/24/36-month trajectories with shaded 95% confidence intervals.
- **Segment Migration:** Sankey diagram showing customer movement between Q1-Q5 quintiles over the last 90 days.
- **Explainability & SLM Box:** 
    - Display SHAP values for the selected customer.
    - **SLM Insight:** A text area showing the SLM-generated "Retention Strategy" (e.g., *"High risk of churn due to session decay; recommend category-specific win-back offer."*).
- **CAC:CLV ROI:** Real-time view of marketing efficiency by acquisition channel.

## 2. Performance & Caching
- **@st.cache_resource**: Load the LightGBM ensemble and LSTM refinement models.
- **@st.cache_data**: Cache the "Gold Layer" feature tables from the Feast offline store.

## 3. UX Standards
- **Hybrid View:** Toggle between "Transactional Value" and "Subscription Value" components.
- **Confidence Gating:** Gray out predictions where the CI width is >75% of the point estimate to prevent low-confidence targeting.