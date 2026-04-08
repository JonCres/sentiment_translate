# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **Subscriber Value Explorer:** Search by `user_id` to view 12/24/36-month revenue curves with component breakdown (SVOD/AVOD/TVOD).
- **Content ROI Heatmap:** Visualization of genre-specific CLTV impact vs. production cost.
- **Explainability & SLM Box:** 
    - Display SHAP waterfall plots for high-value segments.
    - **SLM Narrative:** A text area showing the SLM-generated "Greenlight Report" for proposed content investments based on predicted CLTV lift.
- **Wholesale Partner Analytics:** Comparison of Bundle vs. DTC subscriber value for contract renegotiations.

## 2. Performance & Caching
- **@st.cache_resource**: Load the Tweedie meta-learner and the LSTM sequence model.
- **@st.cache_data**: Cache the "Gold Layer" cohort aggregates from Delta Lake/BigQuery.

## 3. UX Standards
- **Interactivity:** Filter by "Geography," "Acquisition Channel," and "Primary Device."
- **Uncertainty Visualization:** Use shaded confidence intervals for all 12-36 month trajectories; gray out actions where `uncertainty_score` > 0.25.