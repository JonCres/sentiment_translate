# Presentation Layer (Streamlit)

**Role:** Frontend Data Developer (UX & Latency).

## 1. Dashboard Modules
- **Agent Desktop:** Real-time sidebar showing "Live Emotion" and "Recommended Recovery Action" during calls.
- **Product Triage:** Heatmap of SKUs vs. Aspect-Sentiment (Price, Quality, Fit).
- **Explainability & SLM Box:** 
    - Display SHAP values for specific negative interactions.
    - **SLM Summary:** Auto-generated executive briefings for the weekly "Top 10 Pain Points" report.
- **Competitive Pulse:** Comparative metrics showing brand sentiment vs. competitors.

## 2. Performance & Caching
- **@st.cache_resource**: Load heavy Transformer models (BERT/SetFit) and MER fusion weights.
- **@st.cache_data**: Cache the "Gold Layer" interaction aggregates from Delta Lake.

## 3. UX Standards
- **Interactivity:** Filter by "Channel" (Review vs. Call) and "Product Category."
- **Visual Cues:** Use Hawkes Process intensity visualizations to show "Sentiment Momentum" (identifying issues before they go viral).