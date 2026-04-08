import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_json_report, get_ai_interpretation


def render_strategic_intelligence(real_df):
    st.title("🧠 Strategic Intelligence")
    st.markdown("*Deep analysis of drivers, model health, and feature impacts*")
    st.markdown("---")

    correlations = load_json_report("nlp_business_correlations.json")

    # --- Date Filter ---
    if real_df is not None and not real_df.empty and "Timestamp" in real_df.columns:
        dates_series = pd.to_datetime(
            real_df["Timestamp"], errors="coerce"
        ).dt.date.astype(str)
        unique_dates = sorted(dates_series.dropna().unique().tolist(), reverse=True)

        col_filter, _ = st.columns([1, 4])
        with col_filter:
            selected_date = st.selectbox(
                "📅 Filter by Date",
                options=["Overall"] + unique_dates,
                index=0,
                key="strat_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("analysis_drivers", "NPS_Score", type="groq"))

    tabs = st.tabs(["🎯 Business Drivers", "📊 Feature Analysis"])

    with tabs[0]:
        st.subheader("🎯 Key Drivers of Customer Sentiment")
        if correlations and "correlations" in correlations:
            target_metric = st.selectbox(
                "Select Target Metric", list(correlations["correlations"].keys())
            )
            target_corrs = correlations["correlations"][target_metric]

            df_corrs = pd.DataFrame(
                [{"Feature": k, "Correlation": v} for k, v in target_corrs.items()]
            ).sort_values("Correlation", ascending=False)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    df_corrs,
                    x="Correlation",
                    y="Feature",
                    orientation="h",
                    color="Correlation",
                    color_continuous_scale="RdYlGn",
                    template="plotly_white",
                    title=f"Impact on {target_metric}",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 🤖 AI Interpretation")
                interpretation = get_ai_interpretation(
                    "analysis_drivers", target_metric
                )
                st.info(interpretation)
        else:
            st.info("Correlation data not found. Please run the analysis pipeline.")

    with tabs[1]:
        st.subheader("📊 Feature Distributions")
        if real_df is not None:
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Select Feature to Visualize", numeric_cols)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.histogram(
                    real_df,
                    x=selected_col,
                    nbins=30,
                    template="plotly_white",
                    title=f"Distribution of {selected_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 🤖 AI Feature Insight")
                interpretation = get_ai_interpretation(
                    "feature_distributions", selected_col
                )
                st.info(interpretation)

            st.markdown("---")
            st.subheader("🌡️ Feature Correlation Heatmap")
            numeric_df = real_df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Inter-feature Correlations",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🤖 AI Correlation Insight")
                interpretation = get_ai_interpretation("correlation_heatmap")
                st.info(interpretation)
