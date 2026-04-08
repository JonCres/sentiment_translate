import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logic import calculate_overall_health_score
from utils import get_ai_interpretation


def render_temporal_analysis(real_df):
    """Comprehensive time-based analysis of all model outcomes and business metrics."""

    st.title("📅 Temporal Intelligence & Evolution")
    st.markdown("*Track how your customer relationships evolve over time*")
    st.markdown("---")

    if real_df is None or real_df.empty:
        st.warning("No data available for temporal analysis.")
        return

    # Validate timestamp column
    if "Timestamp" not in real_df.columns:
        st.error(
            "Timestamp column not found. Temporal analysis requires date/time data."
        )
        return

    # Convert timestamp to datetime
    df = real_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        st.warning("No valid timestamp data available for analysis.")
        return

    # === TIME RANGE SELECTOR ===
    st.subheader("🎯 Select Analysis Period")

    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="temporal_date_range",
        )

    with col2:
        granularity = st.selectbox(
            "📊 Time Granularity",
            options=["Daily", "Weekly", "Monthly", "Quarterly"],
            index=2,  # Default to Monthly
            key="temporal_granularity",
        )

    with col3:
        # Add comparison toggle
        show_comparison = st.checkbox("📈 Compare Periods", value=False)

    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[
            (df["Timestamp"].dt.date >= start_date)
            & (df["Timestamp"].dt.date <= end_date)
        ]
    else:
        df_filtered = df

    if df_filtered.empty:
        st.warning("No data in selected date range.")
        return

    st.markdown("---")

    # === GROQ AI TEMPORAL INSIGHT ===
    st.markdown("### 🤖 Groq AI Temporal Insight")
    st.info(get_ai_interpretation("temporal_trends", type="groq"))

    st.markdown("---")

    # === TABS FOR DIFFERENT ANALYSES ===
    tabs = st.tabs(
        [
            "📊 Core Metrics Evolution",
            "🎭 Sentiment & Emotion Trends",
            "🏢 Account Health Progression",
            "🎯 Topic & Aspect Dynamics",
            "📈 Predictive Trends",
            "🔬 Advanced Analytics",
        ]
    )

    # Prepare time-grouped data
    time_grouped_df = _prepare_temporal_data(df_filtered, granularity)

    # TAB 1: Core Metrics Evolution
    with tabs[0]:
        _render_core_metrics_evolution(
            df_filtered, time_grouped_df, granularity, show_comparison
        )

    # TAB 2: Sentiment & Emotion Trends
    with tabs[1]:
        _render_sentiment_emotion_trends(df_filtered, time_grouped_df, granularity)

    # TAB 3: Account Health Progression
    with tabs[2]:
        _render_account_health_progression(df_filtered, time_grouped_df, granularity)

    # TAB 4: Topic & Aspect Dynamics
    with tabs[3]:
        _render_topic_aspect_dynamics(df_filtered, time_grouped_df, granularity)

    # TAB 5: Predictive Trends
    with tabs[4]:
        _render_predictive_trends(df_filtered, time_grouped_df, granularity)

    # TAB 6: Advanced Analytics
    with tabs[5]:
        _render_advanced_analytics(df_filtered, time_grouped_df, granularity)


def _prepare_temporal_data(df, granularity):
    """Prepare time-grouped dataframe based on selected granularity."""
    df = df.copy()

    # Create period column based on granularity
    if granularity == "Daily":
        df["Period"] = df["Timestamp"].dt.date
    elif granularity == "Weekly":
        df["Period"] = df["Timestamp"].dt.to_period("W").dt.start_time
    elif granularity == "Monthly":
        df["Period"] = df["Timestamp"].dt.to_period("M").dt.start_time
    else:  # Quarterly
        df["Period"] = df["Timestamp"].dt.to_period("Q").dt.start_time

    return df


def _render_core_metrics_evolution(df, time_df, granularity, show_comparison):
    """Render evolution of core business metrics (NPS, NPS, CSAT, Health Score)."""

    st.subheader("📊 Core Business Metrics Over Time")

    # === METRIC EXPLANATIONS ===
    with st.expander("ℹ️ Understanding Your Metrics", expanded=False):
        st.markdown("""
        **NPS** (-100 to +100)  
        Measures customer loyalty by calculating: (% Promoters - % Detractors)  
        - **Promoters**: Customers who rated 9-10 (highly likely to recommend)
        - **Detractors**: Customers who rated 0-6 (unlikely to recommend)
        - **Passives**: Customers who rated 7-8 (satisfied but not enthusiastic)
        
        **NPS Rating** (0-10 scale)  
        The average of all customer ratings. This is the raw score before calculating the NPS.
        
        **CSAT Score** (1-5 stars)  
        Customer Satisfaction score measuring how satisfied customers are with your service.
        - 5 stars = Very Satisfied
        - 1 star = Very Dissatisfied
        
        **Health Score** (0-100)  
        A composite metric combining NPS and CSAT to give an overall view of customer relationship health.
        - 75-100: Healthy relationships
        - 60-74: Needs attention
        - Below 60: At risk
        """)

    # Group by period and calculate metrics
    def NPS_Index_fn(x):
        promoters = (x >= 9).sum()
        detractors = (x <= 6).sum()
        total = len(x)
        return (promoters - detractors) / total * 100 if total > 0 else 0

    metrics_by_period = (
        time_df.groupby("Period")
        .agg(
            Avg_NPS=("NPS_Score", "mean"),
            NPS_Index=("NPS_Score", NPS_Index_fn),
            Avg_CSAT=("CSAT_Score", "mean"),
            Response_Count=("Interaction_ID", "count"),
        )
        .reset_index()
    )

    # Calculate Health Score for each period using unified logic
    metrics_by_period["Health_Score"] = metrics_by_period.apply(
        lambda row: calculate_overall_health_score(row["NPS_Index"], row["Avg_CSAT"]),
        axis=1,
    )

    # === CHART 1: NPS & HEALTH SCORE (Same -100 to +100 scale) ===
    st.markdown("#### 📈 NPS & Overall Health")

    fig1 = go.Figure()

    # NPS
    fig1.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["NPS_Index"],
            name="NPS",
            line=dict(color="#FF4081", width=4),
            mode="lines+markers",
            marker=dict(size=10),
            hovertemplate="<b>NPS</b>: %{y:.1f}<br>Period: %{x}<extra></extra>",
        )
    )

    # Health Score
    fig1.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["Health_Score"],
            name="Health Score",
            line=dict(color="#66BB6A", width=3),
            mode="lines+markers",
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(102, 187, 106, 0.05)",
            hovertemplate="<b>Health Score</b>: %{y:.1f}<br>Period: %{x}<extra></extra>",
        )
    )

    # Add reference lines
    fig1.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Neutral",
        annotation_position="right",
    )
    fig1.add_hline(
        y=60,
        line_dash="dot",
        line_color="orange",
        annotation_text="Attention Threshold",
        annotation_position="right",
    )

    fig1.update_layout(
        height=450,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.9)",
        font=dict(family="Inter", color="#2c3e50"),
        hovermode="x unified",
        yaxis=dict(title="Score", range=[-100, 100]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig1, use_container_width=True)

    # === CHART 2: NPS & CSAT RATINGS (Separate scales, side by side) ===
    st.markdown("#### ⭐ Customer Satisfaction Ratings")

    col1, col2 = st.columns(2)

    with col1:
        fig_nps = go.Figure()

        fig_nps.add_trace(
            go.Scatter(
                x=metrics_by_period["Period"],
                y=metrics_by_period["Avg_NPS"],
                name="NPS Rating",
                line=dict(color="#00a8cc", width=3),
                mode="lines+markers",
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(0, 168, 204, 0.1)",
                hovertemplate="<b>NPS Rating</b>: %{y:.2f}/10<br>Period: %{x}<extra></extra>",
            )
        )

        # Add reference zones
        fig_nps.add_hrect(
            y0=9,
            y1=10,
            fillcolor="rgba(102, 187, 106, 0.1)",
            line_width=0,
            annotation_text="Promoter Zone",
            annotation_position="top right",
        )
        fig_nps.add_hrect(
            y0=7,
            y1=9,
            fillcolor="rgba(255, 167, 38, 0.1)",
            line_width=0,
            annotation_text="Passive Zone",
            annotation_position="top right",
        )
        fig_nps.add_hrect(
            y0=0,
            y1=7,
            fillcolor="rgba(255, 82, 82, 0.1)",
            line_width=0,
            annotation_text="Detractor Zone",
            annotation_position="top right",
        )

        fig_nps.update_layout(
            title="NPS Rating (0-10 scale)",
            height=400,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            yaxis=dict(title="Rating", range=[0, 10]),
            xaxis=dict(title=""),
            showlegend=False,
        )

        st.plotly_chart(fig_nps, use_container_width=True)

    with col2:
        fig_csat = go.Figure()

        fig_csat.add_trace(
            go.Scatter(
                x=metrics_by_period["Period"],
                y=metrics_by_period["Avg_CSAT"],
                name="CSAT Score",
                line=dict(color="#FFA726", width=3),
                mode="lines+markers",
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(255, 167, 38, 0.1)",
                hovertemplate="<b>CSAT</b>: %{y:.2f}/5 stars<br>Period: %{x}<extra></extra>",
            )
        )

        # Add reference lines
        fig_csat.add_hline(
            y=4,
            line_dash="dot",
            line_color="green",
            annotation_text="Excellent (4+)",
            annotation_position="right",
        )
        fig_csat.add_hline(
            y=3,
            line_dash="dot",
            line_color="orange",
            annotation_text="Good (3+)",
            annotation_position="right",
        )

        fig_csat.update_layout(
            title="CSAT Score (1-5 stars)",
            height=400,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            yaxis=dict(title="Stars", range=[1, 5]),
            xaxis=dict(title=""),
            showlegend=False,
        )

        st.plotly_chart(fig_csat, use_container_width=True)

    # === CHART 3: RESPONSE VOLUME ===
    st.markdown("#### 📊 Feedback Volume")

    fig_volume = go.Figure()

    fig_volume.add_trace(
        go.Bar(
            x=metrics_by_period["Period"],
            y=metrics_by_period["Response_Count"],
            name="Responses",
            marker_color="rgba(0, 168, 204, 0.6)",
            hovertemplate="<b>Responses</b>: %{y}<br>Period: %{x}<extra></extra>",
        )
    )

    fig_volume.update_layout(
        height=300,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.9)",
        yaxis=dict(title="Number of Responses"),
        xaxis=dict(title="Time Period"),
        showlegend=False,
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    # === PERIOD COMPARISON (if enabled) ===
    if show_comparison and len(metrics_by_period) >= 2:
        st.markdown("---")
        st.subheader("📊 Period-over-Period Comparison")

        # Compare most recent vs previous period
        latest = metrics_by_period.iloc[-1]
        previous = metrics_by_period.iloc[-2]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            nps_change = latest["NPS_Index"] - previous["NPS_Index"]
            st.metric(
                "NPS",
                f"{latest['NPS_Index']:.1f}",
                delta=f"{nps_change:+.1f}",
                help="Change in % Promoters - % Detractors",
            )

        with col2:
            nps_change = latest["Avg_NPS"] - previous["Avg_NPS"]
            st.metric(
                "Avg NPS Rating",
                f"{latest['Avg_NPS']:.2f}",
                delta=f"{nps_change:+.2f}",
                help="Change in raw 0-10 average rating",
            )

        with col3:
            csat_change = latest["Avg_CSAT"] - previous["Avg_CSAT"]
            st.metric(
                "Avg CSAT", f"{latest['Avg_CSAT']:.1f}", delta=f"{csat_change:+.1f}"
            )

        with col4:
            health_change = latest["Health_Score"] - previous["Health_Score"]
            st.metric(
                "Health Score",
                f"{latest['Health_Score']:.1f}",
                delta=f"{health_change:+.1f}",
            )

        with col5:
            volume_change = latest["Response_Count"] - previous["Response_Count"]
            volume_pct = (
                (volume_change / previous["Response_Count"] * 100)
                if previous["Response_Count"] > 0
                else 0
            )
            st.metric(
                "Volume",
                int(latest["Response_Count"]),
                delta=f"{volume_pct:+.1f}%",
            )

    # === STATISTICAL SUMMARY ===
    st.markdown("---")
    st.subheader("📈 Statistical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**NPS Statistics**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Avg NPS",
                        "Min NPS",
                        "Max NPS",
                        "Mean NPS Rating",
                        "Std Dev NPS Rating",
                    ],
                    "Value": [
                        f"{metrics_by_period['NPS_Index'].mean():.1f}",
                        f"{metrics_by_period['NPS_Index'].min():.1f}",
                        f"{metrics_by_period['NPS_Index'].max():.1f}",
                        f"{metrics_by_period['Avg_NPS'].mean():.2f}",
                        f"{metrics_by_period['Avg_NPS'].std():.2f}",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("**CSAT & Health Statistics**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Mean CSAT",
                        "Min CSAT",
                        "Max CSAT",
                        "Mean Health Score",
                        "Mean Response Volume",
                    ],
                    "Value": [
                        f"{metrics_by_period['Avg_CSAT'].mean():.2f}",
                        f"{metrics_by_period['Avg_CSAT'].min():.2f}",
                        f"{metrics_by_period['Avg_CSAT'].max():.2f}",
                        f"{metrics_by_period['Health_Score'].mean():.1f}",
                        f"{metrics_by_period['Response_Count'].mean():.1f}",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def _render_sentiment_emotion_trends(df, time_df, granularity):
    """Render sentiment and emotion evolution over time."""

    st.subheader("🎭 Sentiment & Emotional Intelligence Over Time")

    # === SENTIMENT EXPLANATIONS ===
    with st.expander("ℹ️ Understanding Sentiment Metrics", expanded=False):
        st.markdown("""
        **Sentiment Direction** (-1 to +1)  
        Indicates whether feedback is positive or negative:
        - **+1**: Extremely positive language
        - **0**: Neutral language
        - **-1**: Extremely negative language
        
        **Sentiment Strength** (0 to 1)  
        Measures how strongly the sentiment is expressed:
        - **1.0**: Very strong emotion (e.g., "absolutely terrible" or "absolutely amazing")
        - **0.5**: Moderate emotion (e.g., "not great" or "pretty good")
        - **0.0**: Weak or neutral emotion
        
        **Overall Sentiment** (-1 to +1)  
        Combines direction and strength to show the true emotional impact:
        - Strong positive feedback has high positive values
        - Weak positive feedback has low positive values
        - Strong negative feedback has high negative values
        
        **Emotions Detected**  
        Our AI identifies the primary emotion in each piece of feedback:
        - **Joy**: Happy, satisfied, delighted
        - **Neutral**: Factual, objective statements
        - **Sadness**: Disappointed, let down
        - **Anger**: Frustrated, upset
        - **Fear**: Concerned, worried
        - **Surprise**: Unexpected outcomes
        """)

    # === SENTIMENT POLARITY & INTENSITY ===
    if (
        "sentiment_polarity" in time_df.columns
        and "sentiment_intensity" in time_df.columns
    ):
        st.markdown("#### 💬 Customer Sentiment Trends")

        sentiment_by_period = (
            time_df.groupby("Period")
            .agg({"sentiment_polarity": "mean", "sentiment_intensity": "mean"})
            .reset_index()
        )

        # Calculate overall sentiment (polarity * intensity)
        sentiment_by_period["overall_sentiment"] = (
            sentiment_by_period["sentiment_polarity"]
            * sentiment_by_period["sentiment_intensity"]
        )

        # Convert to percentage scales for better readability
        sentiment_by_period["sentiment_direction_pct"] = (
            sentiment_by_period["sentiment_polarity"] * 100
        )
        sentiment_by_period["sentiment_strength_pct"] = (
            sentiment_by_period["sentiment_intensity"] * 100
        )
        sentiment_by_period["overall_sentiment_pct"] = (
            sentiment_by_period["overall_sentiment"] * 100
        )

        # Create two separate charts with appropriate scales
        col1, col2 = st.columns(2)

        with col1:
            fig_direction = go.Figure()

            fig_direction.add_trace(
                go.Scatter(
                    x=sentiment_by_period["Period"],
                    y=sentiment_by_period["sentiment_direction_pct"],
                    name="Sentiment Direction",
                    line=dict(color="#0088cc", width=3),
                    mode="lines+markers",
                    marker=dict(size=8),
                    fill="tozeroy",
                    fillcolor="rgba(0, 136, 204, 0.1)",
                    hovertemplate="<b>Direction</b>: %{y:.1f}%<br>Period: %{x}<extra></extra>",
                )
            )

            # Add reference zones
            fig_direction.add_hrect(
                y0=0,
                y1=100,
                fillcolor="rgba(102, 187, 106, 0.1)",
                line_width=0,
                annotation_text="Positive",
                annotation_position="top right",
            )
            fig_direction.add_hrect(
                y0=-100,
                y1=0,
                fillcolor="rgba(255, 82, 82, 0.1)",
                line_width=0,
                annotation_text="Negative",
                annotation_position="bottom right",
            )
            fig_direction.add_hline(
                y=0, line_dash="dash", line_color="gray", annotation_text="Neutral"
            )

            fig_direction.update_layout(
                title="Sentiment Direction (Positive vs Negative)",
                height=400,
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.9)",
                yaxis=dict(title="Direction (%)", range=[-100, 100]),
                xaxis=dict(title=""),
                showlegend=False,
            )

            st.plotly_chart(fig_direction, use_container_width=True)

        with col2:
            fig_strength = go.Figure()

            fig_strength.add_trace(
                go.Scatter(
                    x=sentiment_by_period["Period"],
                    y=sentiment_by_period["sentiment_strength_pct"],
                    name="Sentiment Strength",
                    line=dict(color="#FFA726", width=3),
                    mode="lines+markers",
                    marker=dict(size=8),
                    fill="tozeroy",
                    fillcolor="rgba(255, 167, 38, 0.1)",
                    hovertemplate="<b>Strength</b>: %{y:.1f}%<br>Period: %{x}<extra></extra>",
                )
            )

            # Add reference lines
            fig_strength.add_hline(
                y=75,
                line_dash="dot",
                line_color="red",
                annotation_text="Very Strong",
                annotation_position="right",
            )
            fig_strength.add_hline(
                y=50,
                line_dash="dot",
                line_color="orange",
                annotation_text="Moderate",
                annotation_position="right",
            )

            fig_strength.update_layout(
                title="Sentiment Strength (How Strongly Expressed)",
                height=400,
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.9)",
                yaxis=dict(title="Strength (%)", range=[0, 100]),
                xaxis=dict(title=""),
                showlegend=False,
            )

            st.plotly_chart(fig_strength, use_container_width=True)

        # Overall Sentiment Chart
        st.markdown("#### 📊 Overall Sentiment (Direction × Strength)")

        fig_overall = go.Figure()

        fig_overall.add_trace(
            go.Scatter(
                x=sentiment_by_period["Period"],
                y=sentiment_by_period["overall_sentiment_pct"],
                name="Overall Sentiment",
                line=dict(color="#66BB6A", width=4),
                mode="lines+markers",
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(102, 187, 106, 0.1)",
                hovertemplate="<b>Overall Sentiment</b>: %{y:.1f}%<br>Period: %{x}<extra></extra>",
            )
        )

        fig_overall.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral",
            annotation_position="right",
        )

        fig_overall.update_layout(
            height=400,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            yaxis=dict(title="Overall Sentiment (%)", range=[-100, 100]),
            xaxis=dict(title="Time Period"),
            hovermode="x unified",
            showlegend=False,
        )

        st.plotly_chart(fig_overall, use_container_width=True)

    st.markdown("---")

    # === EMOTION DISTRIBUTION OVER TIME ===
    if "dominant_emotion" in time_df.columns:
        st.subheader("😊 Emotion Distribution Evolution")

        # Create emotion counts by period
        emotion_pivot = (
            time_df.groupby(["Period", "dominant_emotion"]).size().unstack(fill_value=0)
        )

        # Convert to percentages
        emotion_pct = emotion_pivot.div(emotion_pivot.sum(axis=1), axis=0) * 100

        # Create stacked area chart
        fig = go.Figure()

        # Define vibrant colors for each emotion
        # Include both simple keys and descriptive labels (lowercased) to ensure matching
        emotion_colors = {
            # Joy / Positive
            "joy": "#FFD700",  # Gold
            "positive / content": "#FFD700",
            # Neutral
            "neutral": "#90A4AE",  # Blue Gray
            "neutral / objective": "#90A4AE",
            # Sadness
            "sadness": "#4FC3F7",  # Light Blue
            "disappointed / sad": "#4FC3F7",
            # Anger
            "anger": "#FF5252",  # Red
            "frustrated / angry": "#FF5252",
            # Fear
            "fear": "#9C27B0",  # Purple
            "concerned / worried": "#9C27B0",
            # Surprise
            "surprise": "#FF6F00",  # Deep Orange
            "surprised / uncertain": "#FF6F00",
            # Disgust
            "disgust": "#8D6E63",  # Brown
            "dissatisfied / irritated": "#8D6E63",
        }

        # Add each emotion as a separate trace
        for emotion in emotion_pct.columns:
            color = emotion_colors.get(emotion.lower(), "#CCCCCC")
            fig.add_trace(
                go.Scatter(
                    x=emotion_pct.index,
                    y=emotion_pct[emotion],
                    name=emotion.title(),
                    mode="lines",
                    line=dict(width=0, color=color),
                    fillcolor=color,
                    stackgroup="one",
                    groupnorm="",  # Leave as percentage values
                    hovertemplate=f"<b>{emotion.title()}</b>: %{{y:.1f}}%<extra></extra>",
                )
            )

        fig.update_layout(
            title="Emotion Distribution Over Time (Stacked %)",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=500,
            hovermode="x unified",
            yaxis=dict(title="Percentage", ticksuffix="%", range=[0, 100]),
            xaxis=dict(title="Time Period"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    # === NPS SEGMENT EVOLUTION ===
    st.markdown("---")
    st.subheader("📊 Customer Segment Evolution (NPS)")

    def classify_nps(score):
        if score >= 9:
            return "Promoters"
        elif score >= 7:
            return "Passives"
        else:
            return "Detractors"

    time_df["NPS_Segment"] = time_df["NPS_Score"].apply(classify_nps)
    segment_pivot = (
        time_df.groupby(["Period", "NPS_Segment"]).size().unstack(fill_value=0)
    )

    # Convert to percentages
    segment_pct = segment_pivot.div(segment_pivot.sum(axis=1), axis=0) * 100

    fig = go.Figure()

    segment_colors = {
        "Promoters": "#66BB6A",
        "Passives": "#FFA726",
        "Detractors": "#FF5252",
    }

    for segment in [
        "Detractors",
        "Passives",
        "Promoters",
    ]:  # Order matters for stacking
        if segment in segment_pct.columns:
            fig.add_trace(
                go.Bar(
                    x=segment_pct.index,
                    y=segment_pct[segment],
                    name=segment,
                    marker_color=segment_colors[segment],
                    hovertemplate=f"<b>{segment}</b>: %{{y:.1f}}%<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="stack",
        title="NPS Segment Distribution Over Time",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.9)",
        height=500,
        hovermode="x unified",
        yaxis=dict(title="Percentage", ticksuffix="%"),
        xaxis=dict(title="Time Period"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_account_health_progression(df, time_df, granularity):
    """Render account health metrics progression over time."""

    st.subheader("🏢 Account Health Progression")

    # Calculate health score for each account in each period
    account_health = (
        time_df.groupby(["Period", "Account_Name"])
        .agg({"NPS_Score": "mean", "CSAT_Score": "mean"})
        .reset_index()
    )

    account_health["Health_Score"] = (
        account_health["NPS_Score"] * 10
        + ((account_health["CSAT_Score"] - 1) / 4 * 100)
    ) / 2

    # === TOP/BOTTOM ACCOUNTS TREND ===
    st.markdown("#### 📈 Top & Bottom Performing Accounts")

    # Get latest period health scores
    latest_period = account_health["Period"].max()
    latest_health = account_health[
        account_health["Period"] == latest_period
    ].sort_values("Health_Score")

    top_accounts = latest_health.tail(5)["Account_Name"].tolist()
    bottom_accounts = latest_health.head(5)["Account_Name"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌟 Top 5 Accounts**")
        fig = go.Figure()

        for account in top_accounts:
            account_data = account_health[account_health["Account_Name"] == account]
            fig.add_trace(
                go.Scatter(
                    x=account_data["Period"],
                    y=account_data["Health_Score"],
                    name=account,
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=400,
            yaxis=dict(title="Health Score", range=[0, 100]),
            xaxis=dict(title=""),
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**⚠️ Bottom 5 Accounts**")
        fig = go.Figure()

        for account in bottom_accounts:
            account_data = account_health[account_health["Account_Name"] == account]
            fig.add_trace(
                go.Scatter(
                    x=account_data["Period"],
                    y=account_data["Health_Score"],
                    name=account,
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=400,
            yaxis=dict(title="Health Score", range=[0, 100]),
            xaxis=dict(title=""),
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )

        st.plotly_chart(fig, use_container_width=True)

    # === HEALTH STATUS DISTRIBUTION OVER TIME ===
    st.markdown("---")
    st.markdown("#### 📊 Health Status Distribution Evolution")

    def classify_health(score):
        if score >= 75:
            return "Healthy"
        elif score >= 60:
            return "Needs Attention"
        else:
            return "At Risk"

    account_health["Status"] = account_health["Health_Score"].apply(classify_health)
    status_counts = (
        account_health.groupby(["Period", "Status"]).size().unstack(fill_value=0)
    )

    fig = go.Figure()

    status_colors = {
        "Healthy": "#66BB6A",
        "Needs Attention": "#FFA726",
        "At Risk": "#FF5252",
    }

    for status in ["At Risk", "Needs Attention", "Healthy"]:
        if status in status_counts.columns:
            fig.add_trace(
                go.Bar(
                    x=status_counts.index,
                    y=status_counts[status],
                    name=status,
                    marker_color=status_colors[status],
                )
            )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.9)",
        height=500,
        yaxis=dict(title="Number of Accounts"),
        xaxis=dict(title="Time Period"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_topic_aspect_dynamics(df, time_df, granularity):
    """Render topic and aspect-based sentiment dynamics over time."""

    st.subheader("🎯 Topic & Aspect Dynamics")

    # === TOPIC EVOLUTION ===
    if "Topic_Name" in time_df.columns:
        st.markdown("#### 📊 Topic Prevalence Over Time")

        # Filter out outliers/uncategorized
        topic_df = time_df[~time_df["Topic_Name"].isin(["Outlier", "Uncategorized"])]

        if not topic_df.empty:
            topic_counts = (
                topic_df.groupby(["Period", "Topic_Name"]).size().unstack(fill_value=0)
            )

            # Get top 5 topics overall
            top_topics = topic_df["Topic_Name"].value_counts().head(5).index.tolist()

            fig = go.Figure()

            for topic in top_topics:
                if topic in topic_counts.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=topic_counts.index,
                            y=topic_counts[topic],
                            name=topic,
                            mode="lines+markers",
                            line=dict(width=2),
                            stackgroup="one",
                        )
                    )

            fig.update_layout(
                title="Top 5 Topics Over Time (Stacked)",
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.9)",
                height=500,
                yaxis=dict(title="Mention Count"),
                xaxis=dict(title="Time Period"),
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === ASPECT SENTIMENT EVOLUTION ===
    aspect_cols = [c for c in time_df.columns if c.startswith("sent_")]

    if aspect_cols:
        st.markdown("#### � Aspect-Specific Satisfaction Trends")

        with st.expander("ℹ️ Understanding Aspect Scores", expanded=False):
            st.markdown("""
            **Aspect Scores** (1-5 stars)  
            Our AI analyzes specific aspects of your service mentioned in feedback:
            - **5 stars**: Very satisfied with this aspect
            - **4 stars**: Satisfied
            - **3 stars**: Neutral
            - **2 stars**: Dissatisfied
            - **1 star**: Very dissatisfied
            
            These scores help identify which specific areas are driving satisfaction or dissatisfaction.
            """)

        aspect_trends = time_df.groupby("Period")[aspect_cols].mean()

        fig = go.Figure()

        for col in aspect_trends.columns:
            aspect_name = col.replace("sent_", "").replace("_", " ").title()
            fig.add_trace(
                go.Scatter(
                    x=aspect_trends.index,
                    y=aspect_trends[col],
                    name=aspect_name,
                    mode="lines+markers",
                    line=dict(width=2),
                    hovertemplate=f"<b>{aspect_name}</b>: %{{y:.2f}} stars<br>Period: %{{x}}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Aspect Satisfaction Over Time",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=500,
            yaxis=dict(title="Satisfaction (Stars)", range=[1, 5]),
            xaxis=dict(title="Time Period"),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Add reference lines
        fig.add_hline(
            y=4,
            line_dash="dot",
            line_color="green",
            annotation_text="Satisfied (4+)",
            annotation_position="right",
        )
        fig.add_hline(
            y=3,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral (3)",
            annotation_position="right",
        )

        st.plotly_chart(fig, use_container_width=True)

        # === ASPECT HEATMAP ===
        st.markdown("---")
        st.markdown("#### 🌡️ Aspect Satisfaction Heatmap")

        fig = go.Figure(
            data=go.Heatmap(
                z=aspect_trends.T.values,
                x=aspect_trends.index,
                y=[
                    c.replace("sent_", "").replace("_", " ").title()
                    for c in aspect_trends.columns
                ],
                colorscale="RdYlGn",
                zmid=3,  # Neutral is 3 stars
                zmin=1,
                zmax=5,
                text=aspect_trends.T.values.round(2),
                texttemplate="%{text} ⭐",
                textfont={"size": 10},
                colorbar=dict(title="Stars"),
                hovertemplate="<b>%{y}</b><br>Period: %{x}<br>Rating: %{z:.2f} stars<extra></extra>",
            )
        )

        fig.update_layout(
            title="Aspect Satisfaction Heatmap Over Time",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=400,
            xaxis=dict(title="Time Period"),
            yaxis=dict(title="Aspect"),
        )

        st.plotly_chart(fig, use_container_width=True)


def _render_predictive_trends(df, time_df, granularity):
    """Render predictive model outcomes over time."""

    st.subheader("📈 Predictive Trends & Forecasts")

    # === PREDICTIVE MODEL EXPLANATIONS ===
    with st.expander("ℹ️ Understanding Predictive Metrics", expanded=False):
        st.markdown("""
        **Predicted NPS/CSAT Scores**  
        Our AI models predict what scores customers would give based on their feedback patterns:
        - Helps identify satisfaction levels even when explicit ratings aren't provided
        - Useful for analyzing unstructured feedback
        - **MAE (Mean Absolute Error)**: Shows how accurate our predictions are on average
        
        **Churn Risk** (0-100%)  
        Probability that a customer relationship is at risk:
        - **0-30%**: Low risk - relationship is healthy
        - **30-60%**: Medium risk - needs attention
        - **60-100%**: High risk - immediate action required
        
        This is calculated using:
        - Sentiment trends
        - Satisfaction scores
        - Feedback frequency and recency
        - Topic patterns
        """)

    has_predictions = any(
        col in time_df.columns
        for col in [
            "Predicted_NPS_Score",
            "Predicted_CSAT_Score",
            "Predicted_Churn_Prob",
        ]
    )

    if not has_predictions:
        st.info("Predictive data not available in the current dataset.")
        return

    # === PREDICTED VS ACTUAL ===
    if "Predicted_NPS_Score" in time_df.columns:
        st.markdown("#### 🎯 Predicted vs Actual NPS")

        prediction_comparison = (
            time_df.groupby("Period")
            .agg({"NPS_Score": "mean", "Predicted_NPS_Score": "mean"})
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prediction_comparison["Period"],
                y=prediction_comparison["NPS_Score"],
                name="Actual NPS",
                mode="lines+markers",
                line=dict(color="#0088cc", width=3),
                marker=dict(size=10),
                hovertemplate="<b>Actual NPS</b>: %{y:.2f}/10<br>Period: %{x}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prediction_comparison["Period"],
                y=prediction_comparison["Predicted_NPS_Score"],
                name="Predicted NPS",
                mode="lines+markers",
                line=dict(color="#FFA726", width=3, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="<b>Predicted NPS</b>: %{y:.2f}/10<br>Period: %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=500,
            yaxis=dict(title="NPS Score (0-10)", range=[0, 10]),
            xaxis=dict(title="Time Period"),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate prediction accuracy
        prediction_comparison["Error"] = abs(
            prediction_comparison["NPS_Score"]
            - prediction_comparison["Predicted_NPS_Score"]
        )
        mae = prediction_comparison["Error"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prediction Accuracy (MAE)",
                f"{mae:.2f}",
                help="Mean Absolute Error - Lower is better. This shows the average difference between predicted and actual scores.",
            )
        with col2:
            accuracy_pct = max(0, (1 - mae / 10) * 100)
            st.metric(
                "Model Accuracy",
                f"{accuracy_pct:.1f}%",
                help="Overall prediction accuracy as a percentage",
            )

    # === CHURN RISK EVOLUTION ===
    if "Predicted_Churn_Prob" in time_df.columns:
        st.markdown("---")
        st.markdown("#### ⚠️ Customer Churn Risk Evolution")

        churn_by_period = (
            time_df.groupby("Period")
            .agg({"Predicted_Churn_Prob": "mean"})
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=churn_by_period["Period"],
                y=churn_by_period["Predicted_Churn_Prob"] * 100,
                name="Avg Churn Risk",
                mode="lines+markers",
                line=dict(color="#FF5252", width=3),
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(255, 82, 82, 0.2)",
                hovertemplate="<b>Churn Risk</b>: %{y:.1f}%<br>Period: %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=400,
            yaxis=dict(title="Churn Probability (%)", range=[0, 100]),
            xaxis=dict(title="Time Period"),
            hovermode="x unified",
        )

        # Add risk threshold zones
        fig.add_hrect(
            y0=60,
            y1=100,
            fillcolor="rgba(255, 82, 82, 0.1)",
            line_width=0,
            annotation_text="High Risk",
            annotation_position="top right",
        )
        fig.add_hrect(
            y0=30,
            y1=60,
            fillcolor="rgba(255, 167, 38, 0.1)",
            line_width=0,
            annotation_text="Medium Risk",
            annotation_position="top right",
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning Threshold",
            annotation_position="right",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show current risk level
        latest_risk = churn_by_period.iloc[-1]["Predicted_Churn_Prob"] * 100
        if latest_risk >= 60:
            risk_status = "🔴 High Risk"
            risk_color = "red"
        elif latest_risk >= 30:
            risk_status = "🟡 Medium Risk"
            risk_color = "orange"
        else:
            risk_status = "🟢 Low Risk"
            risk_color = "green"

        st.markdown(
            f"**Current Risk Level:** :{risk_color}[{risk_status} ({latest_risk:.1f}%)]"
        )


def _render_advanced_analytics(df, time_df, granularity):
    """Render advanced temporal analytics and insights."""

    st.subheader("🔬 Advanced Temporal Analytics")

    # === VOLATILITY ANALYSIS ===
    st.markdown("#### 📊 Metric Volatility & Stability")

    volatility_metrics = (
        time_df.groupby("Period")
        .agg({"NPS_Score": ["mean", "std"], "CSAT_Score": ["mean", "std"]})
        .reset_index()
    )

    volatility_metrics.columns = [
        "Period",
        "NPS_Mean",
        "NPS_Std",
        "CSAT_Mean",
        "CSAT_Std",
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=volatility_metrics["Period"],
                y=volatility_metrics["NPS_Std"],
                name="NPS Volatility",
                mode="lines+markers",
                line=dict(color="#0088cc", width=3),
                fill="tozeroy",
                fillcolor="rgba(0, 136, 204, 0.2)",
            )
        )

        fig.update_layout(
            title="NPS Volatility (Standard Deviation)",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=350,
            yaxis=dict(title="Std Dev"),
            xaxis=dict(title=""),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=volatility_metrics["Period"],
                y=volatility_metrics["CSAT_Std"],
                name="CSAT Volatility",
                mode="lines+markers",
                line=dict(color="#FFA726", width=3),
                fill="tozeroy",
                fillcolor="rgba(255, 167, 38, 0.2)",
            )
        )

        fig.update_layout(
            title="CSAT Volatility (Standard Deviation)",
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0.9)",
            height=350,
            yaxis=dict(title="Std Dev"),
            xaxis=dict(title=""),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === MOVING AVERAGES ===
    st.markdown("#### 📈 Trend Analysis (Moving Averages)")

    window_size = st.slider(
        "Moving Average Window",
        min_value=2,
        max_value=10,
        value=3,
        help="Number of periods to average",
    )

    metrics_by_period = (
        time_df.groupby("Period")
        .agg({"NPS_Score": "mean", "CSAT_Score": "mean"})
        .reset_index()
    )

    metrics_by_period["NPS_MA"] = (
        metrics_by_period["NPS_Score"].rolling(window=window_size).mean()
    )
    metrics_by_period["CSAT_MA"] = (
        metrics_by_period["CSAT_Score"].rolling(window=window_size).mean()
    )

    fig = make_subplots(rows=2, cols=1, subplot_titles=("NPS Trend", "CSAT Trend"))

    # NPS
    fig.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["NPS_Score"],
            name="NPS (Actual)",
            mode="markers",
            marker=dict(color="#0088cc", size=6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["NPS_MA"],
            name=f"NPS ({window_size}-period MA)",
            mode="lines",
            line=dict(color="#0088cc", width=3),
        ),
        row=1,
        col=1,
    )

    # CSAT
    fig.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["CSAT_Score"],
            name="CSAT (Actual)",
            mode="markers",
            marker=dict(color="#FFA726", size=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics_by_period["Period"],
            y=metrics_by_period["CSAT_MA"],
            name=f"CSAT ({window_size}-period MA)",
            mode="lines",
            line=dict(color="#FFA726", width=3),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.9)",
        showlegend=True,
    )

    fig.update_xaxes(title_text="Time Period", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # === CORRELATION ANALYSIS ===
    st.markdown("#### 🔗 Cross-Metric Correlation Over Time")

    if "sentiment_polarity" in time_df.columns:
        correlation_data = (
            time_df.groupby("Period")
            .agg(
                {
                    "NPS_Score": "mean",
                    "CSAT_Score": "mean",
                    "sentiment_polarity": "mean",
                }
            )
            .reset_index()
        )

        # Calculate rolling correlation between NPS and sentiment polarity
        window = min(5, len(correlation_data) - 1)
        if window > 1:
            correlation_data["NPS_Sentiment_Corr"] = (
                correlation_data["NPS_Score"]
                .rolling(window=window)
                .corr(correlation_data["sentiment_polarity"])
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=correlation_data["Period"],
                    y=correlation_data["NPS_Sentiment_Corr"],
                    name="NPS vs Sentiment Correlation",
                    mode="lines+markers",
                    line=dict(color="#9C27B0", width=3),
                )
            )

            fig.update_layout(
                title=f"Rolling Correlation (window={window}): NPS vs Sentiment Polarity",
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.9)",
                height=400,
                yaxis=dict(title="Correlation Coefficient", range=[-1, 1]),
                xaxis=dict(title="Time Period"),
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            st.plotly_chart(fig, use_container_width=True)
