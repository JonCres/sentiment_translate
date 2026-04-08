import streamlit as st
import pandas as pd
import plotly.express as px
from config import NPS_COLORSCALE, NPS_COLOR_RANGE
from logic import get_top_themes
from utils import get_ai_interpretation
from components.sidebar import navigate_to_topic


def render_key_insights(real_df):
    st.title("💡 Key Insights & Emerging Themes")
    st.markdown("*Discover what customers are talking about*")
    st.markdown("---")

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
                key="key_insights_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    themes = get_top_themes(real_df)

    # Marketing Insights Summary
    st.subheader("🎯 Marketing Intelligence Summary")

    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("analysis_drivers", "CSAT_Score", type="groq"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌟 High-Sentiment Topics")

        if themes["positive"]:
            for theme in themes["positive"]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 1.1rem; font-weight: 600;">✨ {theme["theme"]}</span>
                                <span style="color: #66BB6A; font-weight: 700; font-size: 1.2rem;">{theme["score"]} NPS</span>
                            </div>
                            <p style="margin-top: 8px; color: #5a6c7d; font-size: 0.9rem;">
                                <strong>{theme["count"]} mentions.</strong> Strength to leverage in marketing and sales collateral.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"🔍 View {theme['count']} mentions",
                        key=f"btn_pos_{theme['id']}",
                        on_click=navigate_to_topic,
                        args=(theme["theme"],),
                    )
        else:
            st.info("Analyzing customer feedback for positive topics...")

    with col2:
        st.markdown("### ⚠️ Low-Sentiment Topics")

        if themes["negative"]:
            for theme in themes["negative"]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 1.1rem; font-weight: 600;">⚡ {theme["theme"]}</span>
                                <span style="color: #FF5252; font-weight: 700; font-size: 1.2rem;">{theme["score"]} NPS</span>
                            </div>
                            <p style="margin-top: 8px; color: #5a6c7d; font-size: 0.9rem;">
                                <strong>{theme["count"]} mentions.</strong> Priority area for product improvement and messaging refinement.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"🔍 View {theme['count']} mentions",
                        key=f"btn_neg_{theme['id']}",
                        on_click=navigate_to_topic,
                        args=(theme["theme"],),
                    )
        else:
            st.success("No major negative topics identified!")

    st.markdown("---")

    # Topic Analysis
    if real_df is not None and "Topic_Name" in real_df.columns:
        st.subheader("📊 Topic Distribution & Trends")

        # Filter out outliers if needed
        topic_df = (
            real_df[real_df["Topic_Name"] != "Outlier"]
            if "Topic_Name" in real_df.columns
            else real_df
        )

        if not topic_df.empty:
            topic_counts = topic_df["Topic_Name"].value_counts().head(10).reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig = px.treemap(
                topic_counts,
                path=["Topic"],
                values="Count",
                color="Count",
                color_continuous_scale="Viridis",
                template="plotly_white",
            )

            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0.9)",
                font=dict(family="Inter", color="#2c3e50"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Topic vs Sentiment Analysis
            st.markdown("---")
            st.subheader("🎭 How Sentiment Varies by Topic")

            if "NPS_Score" in real_df.columns:
                topic_sentiment = (
                    topic_df.groupby("Topic_Name")["NPS_Score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )

                fig = px.bar(
                    topic_sentiment,
                    x="Topic_Name",
                    y="NPS_Score",
                    color="NPS_Score",
                    color_continuous_scale=NPS_COLORSCALE,
                    range_color=NPS_COLOR_RANGE,
                    template="plotly_white",
                    text="NPS_Score",
                )

                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")

                fig.update_layout(
                    paper_bgcolor="rgba(255,255,255,0.9)",
                    plot_bgcolor="rgba(248,249,250,1)",
                    xaxis_title="Topic",
                    yaxis_title="Average NPS Score",
                    yaxis_range=[0, 10],
                    font=dict(family="Inter", color="#2c3e50"),
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)
