import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import NPS_COLORSCALE, NPS_COLOR_RANGE
from logic import get_sentiment_insights
from utils import get_ai_interpretation


def render_customer_sentiment(real_df):
    st.title("📈 Customer Sentiment Analysis")
    st.markdown("*Understanding how customers feel about your brand*")

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
                key="cust_sent_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    with st.expander("ℹ️ Deep Dive: Context-Aware Sentiment Model"):
        st.markdown("""
        Beyond simple keyword matching, our **Context-Aware Sentiment Model** (based on BERT/RoBERTa) analyzes the linguistic structure and professional tone of B2B feedback to extract three granular metrics:
        
        1.  **Polarity (-1.0 to +1.0):** Measures the direction of sentiment. 
            *   *Values near +1.0* indicate high-conviction positive feedback.
            *   *Values near -1.0* indicate severe dissatisfaction or implicit criticism.
        2.  **Intensity (0.0 to 1.0):** Measures the confidence or "strength" of the expressed emotion. A low score indicates a lukewarm or neutral tone, while a high score indicates passionate feedback.
        3.  **Context Score (Combined):** The mathematical product of *Polarity x Intensity*. It highlights the "signals that matter" by amplifying high-intensity feedback and dampening neutral noise.
        """)
        st.latex(r"\text{Context Score} = \text{Polarity} \times \text{Intensity}")

    st.markdown("---")

    # Nuanced Sentiment Metrics
    st.subheader("🧠 Context-Aware Sentiment Intelligence")
    sentiment_data = get_sentiment_insights(real_df)

    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("correlation_heatmap", type="groq"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Average Polarity",
            sentiment_data["avg_polarity"],
            help="Nuanced sentiment from -1 (Negative) to +1 (Positive). Takes B2B context into account.",
        )
    with c2:
        st.metric(
            "Average Intensity",
            sentiment_data["avg_intensity"],
            help="Confidence/Strength of the expressed sentiment (0 to 1).",
        )
    with c3:
        st.metric(
            "Context Score",
            round(sentiment_data["avg_polarity"] * sentiment_data["avg_intensity"], 2),
            help="Combined metric: Polarity x Intensity. High values indicate high-conviction positive feedback.",
        )

    if real_df is not None and "sentiment_polarity" in real_df.columns:
        st.markdown("#### 🗺️ Sentiment Landscape")
        fig = px.scatter(
            real_df,
            x="sentiment_polarity",
            y="sentiment_intensity",
            color="NPS_Score",
            hover_data=["Account_Name", "Interaction_Payload"],
            color_continuous_scale=NPS_COLORSCALE,
            range_color=NPS_COLOR_RANGE,
            template="plotly_white",
            labels={
                "sentiment_polarity": "Polarity (Neg to Pos)",
                "sentiment_intensity": "Intensity (Confidence)",
            },
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="NPS Score",
                tickvals=[0, 2, 4, 6, 8, 10],
                ticktext=["0 (Det)", "2", "4", "6", "8", "10 (Pro)"],
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    if real_df is not None and not real_df.empty:
        # Sentiment over time
        if "Timestamp" in real_df.columns and pd.notna(real_df["Timestamp"]).any():
            st.subheader("📅 Sentiment Trend Over Time")

            df_time = real_df.copy()
            df_time["Timestamp"] = pd.to_datetime(df_time["Timestamp"])
            df_time["Month"] = df_time["Timestamp"].dt.to_period("M").astype(str)

            monthly_nps = df_time.groupby("Month")["NPS_Score"].mean().reset_index()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=monthly_nps["Month"],
                    y=monthly_nps["NPS_Score"],
                    mode="lines+markers",
                    name="NPS Trend",
                    line=dict(color="#0088cc", width=3),
                    marker=dict(size=10, color="#0088cc"),
                    fill="tozeroy",
                    fillcolor="rgba(0, 136, 204, 0.2)",
                )
            )

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.9)",
                plot_bgcolor="rgba(248,249,250,1)",
                xaxis_title="Month",
                yaxis_title="Average NPS Score",
                font=dict(family="Inter", color="#2c3e50"),
                hovermode="x unified",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # NPS score by account tier
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 NPS score by Account Tier")

            if "Tier" in real_df.columns and "NPS_Score" in real_df.columns:
                tier_nps = (
                    real_df.groupby("Tier")["NPS_Score"]
                    .mean()
                    .sort_values(ascending=True)
                    .reset_index()
                )

                fig = px.bar(
                    tier_nps,
                    x="NPS_Score",
                    y="Tier",
                    orientation="h",
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
                    showlegend=False,
                    xaxis_title="Average NPS Score",
                    xaxis_range=[0, 10],
                    yaxis_title="",
                    font=dict(family="Inter", color="#2c3e50"),
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("😊 Emotion Distribution")

            if "dominant_emotion" in real_df.columns:
                emo_counts = real_df["dominant_emotion"].value_counts().head(6)

                fig = px.pie(
                    values=emo_counts.values,
                    names=emo_counts.index,
                    hole=0.5,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )

                fig.update_layout(
                    paper_bgcolor="rgba(255,255,255,0.9)",
                    font=dict(family="Inter", color="#2c3e50"),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5),
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Aspect sentiment analysis
        st.subheader("🎯 What Customers Love & What Needs Work")

        with st.expander("ℹ️ How are these scores calculated?"):
            st.markdown("""
            These scores are derived from our **Aspect-Based Sentiment Analysis (ABSA)** model. 
            The raw 1-5 star ratings are normalized to a **0-100% Sentiment Strength** scale:
            *   **100%:** Perfect 5-star sentiment.
            *   **50%:** Neutral (3-star) sentiment.
            *   **0%:** Extremely negative (1-star) sentiment.
            """)

        aspect_cols = [c for c in real_df.columns if c.startswith("sent_")]

        if aspect_cols:
            # Calculate raw means
            avg_aspects = real_df[aspect_cols].mean().sort_values()

            # Normalize to 0-100%
            normalized_aspects = ((avg_aspects - 1) / 4.0 * 100).round(1)
            normalized_aspects.index = [
                c.replace("sent_", "").title() for c in normalized_aspects.index
            ]

            # Split into positive and negative based on the 50% (3-star) threshold
            positive_aspects = normalized_aspects[normalized_aspects >= 50].sort_values(
                ascending=False
            )
            negative_aspects = normalized_aspects[normalized_aspects < 50].sort_values()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ Strengths")

                if len(positive_aspects) > 0:
                    fig = px.bar(
                        x=positive_aspects.values,
                        y=positive_aspects.index,
                        orientation="h",
                        color_discrete_sequence=["#66BB6A"],
                        template="plotly_dark",
                        labels={"x": "Sentiment Strength (%)", "y": ""},
                    )

                    fig.update_layout(
                        paper_bgcolor="rgba(255,255,255,0.9)",
                        plot_bgcolor="rgba(248,249,250,1)",
                        xaxis_range=[0, 100],
                        font=dict(family="Inter", color="#2c3e50"),
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strong positive aspects identified yet.")

            with col2:
                st.markdown("### ⚠️ Areas for Improvement")

                if len(negative_aspects) > 0:
                    fig = px.bar(
                        x=negative_aspects.values,
                        y=negative_aspects.index,
                        orientation="h",
                        color_discrete_sequence=["#FF5252"],
                        template="plotly_dark",
                        labels={"x": "Sentiment Strength (%)", "y": ""},
                    )

                    fig.update_layout(
                        paper_bgcolor="rgba(255,255,255,0.9)",
                        plot_bgcolor="rgba(248,249,250,1)",
                        xaxis_range=[0, 100],
                        font=dict(family="Inter", color="#2c3e50"),
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("All aspects performing well!")
