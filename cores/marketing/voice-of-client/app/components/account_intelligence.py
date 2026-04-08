import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import get_ai_interpretation, load_params
from logic import get_portfolio_data


def render_account_intelligence(real_df):
    st.title("🎯 Account Intelligence")
    st.markdown("*Deep dive into individual account health and engagement*")
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
                key="acc_int_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    # Groq Account Insight - shown at top, always visible
    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("model_evaluation", type="groq"))

    st.markdown("---")

    portfolio = get_portfolio_data(real_df)

    # Account selector with search
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_account = st.selectbox(
            "🔍 Select Account",
            portfolio["Account Name"].tolist() if not portfolio.empty else [],
            help="Choose an account to view detailed insights",
        )

    if selected_account and not portfolio.empty:
        acc_data = portfolio[portfolio["Account Name"] == selected_account].iloc[0]

        with col2:
            tier_emoji = (
                "👑"
                if acc_data["Tier"] == "Premium"
                else "⭐"
                if acc_data["Tier"] == "Enterprise"
                else "📊"
            )
            st.metric("Account Tier", f"{tier_emoji} {acc_data['Tier']}")

        with col3:
            st.metric("Feedback Volume", acc_data["Feedback Count"])

        st.markdown("---")
        st.subheader(f"📊 {selected_account} - Health Dashboard")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            health_color = acc_data["Color"]
            st.markdown(
                f"""
                <div title="Composite metric (0-100) representing overall relationship stability, derived from aggregated NPS and CSAT scores." 
                     style="text-align: center; padding: 20px; background: linear-gradient(135deg, {health_color}20 0%, {health_color}10 100%); border-radius: 12px; border: 2px solid {health_color}; cursor: help;">
                    <h3 style="margin: 0; color: {health_color};">{acc_data["Health Score"]}/100</h3>
                    <p style="margin: 8px 0 0 0; color: #5a6c7d;">Health Score</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            nps_color = (
                "#66BB6A"
                if acc_data["Avg NPS"] >= 7
                else "#FFA726"
                if acc_data["Avg NPS"] >= 0
                else "#FF5252"
            )
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {nps_color}20 0%, {nps_color}10 100%); border-radius: 12px; border: 2px solid {nps_color};">
                    <h3 style="margin: 0; color: {nps_color};">{acc_data["Avg NPS"]}</h3>
                    <p style="margin: 8px 0 0 0; color: #5a6c7d;">NPS Score</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            csat_color = (
                "#66BB6A"
                if acc_data["Avg CSAT"] >= 4
                else "#FFA726"
                if acc_data["Avg CSAT"] >= 3
                else "#FF5252"
            )
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, {csat_color}20 0%, {csat_color}10 100%); border-radius: 12px; border: 2px solid {csat_color};">
                    <h3 style="margin: 0; color: {csat_color};">{acc_data["Avg CSAT"]}/5</h3>
                    <p style="margin: 8px 0 0 0; color: #5a6c7d;">CSAT Score</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            status_color = acc_data["Color"]
            st.markdown(
                f"""
                <div title="Current health category based on thresholds: Healthy (>75), Needs Attention (60-75), or At Risk (<60)." 
                     style="text-align: center; padding: 20px; background: linear-gradient(135deg, {status_color}20 0%, {status_color}10 100%); border-radius: 12px; border: 2px solid {status_color}; cursor: help;">
                    <h3 style="margin: 0; color: {status_color}; font-size: 1.3rem;">{acc_data["Status"]}</h3>
                    <p style="margin: 8px 0 0 0; color: #5a6c7d;">Status</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # New: Nuanced Sentiment row for account
        if real_df is not None:
            acc_feedback = real_df[real_df["Account_Name"] == selected_account]
            if not acc_feedback.empty and "sentiment_polarity" in acc_feedback.columns:
                st.markdown("<br>", unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                avg_p = round(acc_feedback["sentiment_polarity"].mean(), 2)
                avg_i = round(acc_feedback["sentiment_intensity"].mean(), 2)

                s1.metric(
                    "Account Polarity",
                    avg_p,
                    help="Continuous sentiment scale (-1 to 1)",
                )
                s2.metric(
                    "Response Intensity",
                    avg_i,
                    help="Conviction/Strength of feedback (0 to 1)",
                )
                s3.metric(
                    "Nuanced Health",
                    round(avg_p * avg_i, 2),
                    help="Combined contextual score",
                )

        st.markdown("---")

        # Sentiment breakdown for account
        if real_df is not None:
            acc_feedback = real_df[real_df["Account_Name"] == selected_account]

            if not acc_feedback.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎭 Emotional Intelligence")

                    if "dominant_emotion" in acc_feedback.columns:
                        emo_dist = acc_feedback["dominant_emotion"].value_counts()

                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=emo_dist.index,
                                    values=emo_dist.values,
                                    hole=0.5,
                                    marker=dict(colors=px.colors.qualitative.Set3),
                                    textinfo="label+percent",
                                    textfont=dict(size=14, color="#2c3e50"),
                                )
                            ]
                        )

                        fig.update_layout(
                            template="plotly_white",
                            paper_bgcolor="rgba(255,255,255,0.9)",
                            font=dict(family="Inter", color="#2c3e50"),
                            showlegend=True,
                            height=400,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("💬 Recent Sentiment")

                    if (
                        "Timestamp" in acc_feedback.columns
                        and "NPS_Score" in acc_feedback.columns
                    ):
                        recent_feedback = acc_feedback.sort_values(
                            "Timestamp", ascending=False
                        ).head(5)

                        for _, row in recent_feedback.iterrows():
                            sentiment_emoji = (
                                "😊"
                                if row["NPS_Score"] >= 9
                                else "😐"
                                if row["NPS_Score"] >= 7
                                else "😞"
                            )
                            params = load_params()
                            ts_format = (
                                params.get("data_processing", {})
                                .get("skeleton_mapping", {})
                                .get("defaults", {})
                                .get("Timestamp_Format", "%Y-%m-%d")
                            )

                            ts_val = row["Timestamp"]
                            if pd.notna(ts_val):
                                try:
                                    date_str = pd.to_datetime(ts_val).strftime(
                                        ts_format
                                    )
                                except:
                                    date_str = str(ts_val)[:10]
                            else:
                                date_str = "Unknown"

                            st.markdown(
                                f"""
                                <div class="customer-quote">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                        <span style="font-weight: 600;">{sentiment_emoji} NPS: {row["NPS_Score"]}</span>
                                        <span style="color: #5a6c7d; font-size: 0.85rem;">{date_str}</span>
                                    </div>
                                    <p style="margin: 0; font-size: 0.9rem;">"{row.get("Interaction_Payload", "No comment")[:150]}..."</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.markdown("---")

                # Aspect sentiment radar for this account
                st.subheader("📡 Sentiment Across Key Aspects")

                aspect_cols = [c for c in acc_feedback.columns if c.startswith("sent_")]

                if aspect_cols:
                    aspect_scores = acc_feedback[aspect_cols].mean()
                    aspect_names = [
                        c.replace("sent_", "").title() for c in aspect_scores.index
                    ]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatterpolar(
                            r=aspect_scores.values,
                            theta=aspect_names,
                            fill="toself",
                            name=selected_account,
                            line=dict(color="#0088cc", width=2),
                            fillcolor="rgba(0, 136, 204, 0.25)",
                        )
                    )

                    # Add portfolio average for comparison
                    if len(real_df) > len(acc_feedback):
                        portfolio_avg = real_df[aspect_cols].mean()

                        fig.add_trace(
                            go.Scatterpolar(
                                r=portfolio_avg.values,
                                theta=aspect_names,
                                fill="toself",
                                name="Portfolio Average",
                                line=dict(color="#FFA726", width=2, dash="dash"),
                                fillcolor="rgba(255, 167, 38, 0.1)",
                            )
                        )

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, range=[1, 5], gridcolor="#d1d9e0"
                            ),
                            angularaxis=dict(gridcolor="#d1d9e0"),
                        ),
                        showlegend=True,
                        template="plotly_white",
                        paper_bgcolor="rgba(255,255,255,0.9)",
                        font=dict(family="Inter", color="#2c3e50"),
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)
