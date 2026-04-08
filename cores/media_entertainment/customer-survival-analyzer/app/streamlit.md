# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **Subscriber Survival Explorer:** Search by `subscriber_id` to view individual Hazard Functions and P(Survival) curves.
- **Cohort Benchmarking:** Interactive Kaplan-Meier curves comparing acquisition channels (e.g., Paid Search vs. Organic).
- **Explainability & SLM Box:** 
    - Display SHAP values for high-risk subscribers.
    - **SLM Summary:** Auto-generated natural language briefings for the weekly "Retention Triage" report.
- **Intervention Window Tracker:** Visual timeline showing the "Optimal Contact Window" (14-21 days pre-termination).

## 2. Performance & Caching
- **@st.cache_resource**: Load the DeepSurv neural network and SHAP explainer.
- **@st.cache_data**: Cache aggregated survival probabilities and Brier scores from the Kedro Feature Store.

## 3. UX Standards
- **Interactivity:** Filter by "Risk Segment" (Very High to Very Low) and "Plan Tier."
- **Visual Cues:** Use red/orange color coding for hazard spikes; display 95% Confidence Intervals for all median tenure predictions.