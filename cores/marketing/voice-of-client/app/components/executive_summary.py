import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from logic import (
    get_sentiment_insights,
    get_portfolio_data,
    generate_word_clouds,
    calculate_overall_health_score,
    calculate_temporal_deltas,
)
from utils import get_ai_interpretation
from components.sidebar import navigate_to_page


def render_executive_summary(real_df):
    st.title("🏠 Executive Summary")
    st.markdown("*Your customer health snapshot at a glance*")
    st.markdown("---")

    # Store original unfiltered dataframe for temporal comparisons
    original_df = real_df.copy() if real_df is not None else None
    selected_date = "Overall"

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
                key="exec_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    with st.expander("ℹ️ What is the Portfolio NPS?"):
        st.markdown("""
        The **Portfolio NPS** (based on the Net Promoter Score methodology) is a widely used market research metric. It represents the aggregate "health" of your customer relationships.
        
        ### The Formula
        The Index is calculated by subtracting the percentage of customers who are Detractors from the percentage who are Promoters.
        """)
        st.latex(r"\text{NPS} = \% \text{Promoters} - \% \text{Detractors}")
        st.markdown("""
        ### Customer Categories
        *   **😊 Promoters (9-10):** Loyal enthusiasts who will keep buying and refer others.
        *   **😐 Passives (7-8):** Satisfied but unenthusiastic customers.
        *   **😞 Detractors (0-6):** Unhappy customers who can damage your brand.
        
        The index can range from **-100** (critical risk) to **+100** (world-class loyalty).
        """)

    with st.expander("ℹ️ What is Customer Satisfaction (CSAT)?"):
        st.markdown("""
        **Customer Satisfaction (CSAT)** measures a customer's satisfaction with a specific interaction, product, or service. In this dashboard, it represents the average rating provided by clients.
        
        ### The Formula
        Our CSAT is calculated as the arithmetic mean of all satisfaction ratings (on a 1-5 scale).
        """)
        st.latex(
            r"CSAT = \frac{\sum \text{Satisfaction Ratings}}{\text{Total Responses}}"
        )
        st.markdown("""
        ### Rating Scale
        *   **5 - Extremely Satisfied:** The service exceeded all expectations.
        *   **4 - Satisfied:** The service met expectations.
        *   **3 - Neutral:** The service was acceptable but not remarkable.
        *   **2 - Dissatisfied:** The service failed to meet some expectations.
        *   **1 - Extremely Dissatisfied:** Major issues were encountered.
        """)

    with st.expander("ℹ️ What is the Health Score?"):
        st.markdown("""
        The **Health Score** is a composite metric (0-100) that combines NPS and CSAT to provide a holistic view of account relationship stability.
        
        ### The Formula
        """)
        st.latex(
            r"\text{Health Score} = \frac{(\text{NPS}_{normalized} + \text{CSAT}_{normalized})}{2}"
        )
        st.markdown("""
        Where:
        - **NPS Normalized** = NPS Score × 10 (scales 0-10 to 0-100)
        - **CSAT Normalized** = (CSAT - 1) / 4 × 100 (scales 1-5 to 0-100)
        """)
        st.info("""
        **🎯 Strategic Usage:** The primary validity of combining these metrics lies in **Predictive Power**. 
        
        While NPS is often a *lagging indicator* of past experience, CSAT can act as a *leading indicator* of sentiment shifts. 
        Integrating them allows for **'SuccessPlays'**—automated interventions that trigger if, for example, a high-value account 
        shows a 'Promoter' NPS but a sharp decline in CSAT across recent support interactions.
        
        This early-warning system helps Customer Success teams prioritize proactive outreach before relationship erosion occurs.
        """)

    # Calculate temporal deltas using ORIGINAL unfiltered data
    temporal_deltas = calculate_temporal_deltas(
        original_df,
        selected_date if selected_date != "Overall" else None,
    )

    # Key Metrics Row - using FILTERED data for current values
    sentiment_data = get_sentiment_insights(real_df)
    portfolio = get_portfolio_data(real_df)

    avg_csat = portfolio["Avg CSAT"].mean() if not portfolio.empty else 0
    overall_health_score = calculate_overall_health_score(
        sentiment_data.get("nps", 0), avg_csat
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Dynamic health score delta calculation
        health_delta = temporal_deltas.get("health_delta")
        health_score_delta = (
            f"{health_delta:+.1f}" if health_delta is not None else None
        )
        st.metric(
            "Overall Health Score",
            overall_health_score,
            delta=health_score_delta,
            help="Composite score combining NPS and CSAT, scaled 0-100.",
        )

    with col2:
        # Dynamic NPS delta calculation
        nps_delta_val = temporal_deltas.get("nps_delta")
        nps_delta = f"{nps_delta_val:+.1f}" if nps_delta_val is not None else None
        st.metric(
            "Portfolio NPS",
            sentiment_data.get("nps", 0),
            delta=nps_delta,
            help="Computed as % Promoters - % Detractors. Range: -100 to +100.",
        )

    with col3:
        # Dynamic CSAT delta calculation
        csat_delta_val = temporal_deltas.get("csat_delta")
        csat_delta = f"{csat_delta_val:+.1f}" if csat_delta_val is not None else None
        st.metric(
            "Customer Satisfaction",
            f"{round(avg_csat, 1)}/5",
            delta=csat_delta,
            help="Average satisfaction rating on a scale of 1-5.",
        )

    with col4:
        total_accounts = len(portfolio)
        st.metric(
            "Active Accounts",
            total_accounts,
            delta=f"{len(portfolio[portfolio['Status'] == 'Healthy'])} healthy",
        )

    with col5:
        total_feedback = sentiment_data.get("total", 0)
        # Dynamic feedback count delta calculation
        feedback_delta_pct = temporal_deltas.get("feedback_delta_pct")
        if feedback_delta_pct is not None:
            feedback_delta = f"{feedback_delta_pct:+.1f}% vs prev period"
        else:
            feedback_delta = None
        st.metric("Total Responses", total_feedback, delta=feedback_delta)

    st.markdown("---")

    # Groq Insight for Executive Summary
    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("analysis_drivers", "NPS_Score", type="groq"))

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 NPS")

        if sentiment_data and sentiment_data["total"] > 0:
            # Calculate reference value for gauge delta (previous period NPS)
            nps_delta_val = temporal_deltas.get("nps_delta")
            current_nps = sentiment_data["nps"]
            reference_nps = (
                (current_nps - nps_delta_val)
                if nps_delta_val is not None
                else current_nps
            )

            # Create a more visually appealing gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment_data["nps"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": "NPS",
                        "font": {"size": 24, "color": "#FFFFFF"},
                    },
                    delta={
                        "reference": reference_nps,
                        "increasing": {"color": "#66BB6A"},
                    },
                    gauge={
                        "axis": {
                            "range": [-100, 100],
                            "tickwidth": 1,
                            "tickcolor": "#90A4AE",
                        },
                        "bar": {"color": "#00D9FF"},
                        "bgcolor": "rgba(255,255,255,0.1)",
                        "borderwidth": 2,
                        "bordercolor": "#2B3240",
                        "steps": [
                            {"range": [-100, 0], "color": "rgba(255, 82, 82, 0.3)"},
                            {"range": [0, 50], "color": "rgba(255, 167, 38, 0.3)"},
                            {"range": [50, 100], "color": "rgba(102, 187, 106, 0.3)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.75,
                            "value": 70,
                        },
                    },
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0.9)",
                plot_bgcolor="rgba(255,255,255,0.9)",
                font={"color": "#2c3e50", "family": "Inter"},
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Understanding the Gauge:**
                *   **Main Number:** Represents the current **Portfolio NPS**, ranging from -100 to +100.
                *   **Delta (under Index):** Shows the change relative to the previous period.
                *   **Color Zones:**
                    *   <span style="color:red">**Red (-100 to 0):**</span> High proportion of Detractors (critical risk).
                    *   <span style="color:orange">**Orange (0 to 50):**</span> Passive sentiment mix.
                    *   <span style="color:green">**Green (50 to 100):**</span> Strong Promoter base (loyalty).
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("📊 Customer Sentiment Distribution")

        if sentiment_data and sentiment_data["total"] > 0:
            passive_pct = round(
                sentiment_data["passives"] / sentiment_data["total"] * 100, 1
            )
            st.markdown(
                f"""
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <span style="color: #2E7D32; font-weight: 600;">😊 Promoters</span>
                        <span style="color: #2c3e50; font-weight: 700;">{sentiment_data["promoter_pct"]}%</span>
                    </div>
                    <div style="width: 100%; background: #e9ecef; border-radius: 4px; height: 8px;">
                        <div style="width: {sentiment_data["promoter_pct"]}%; background: #66BB6A; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <span style="color: #EF6C00; font-weight: 600;">😐 Passives</span>
                        <span style="color: #2c3e50; font-weight: 700;">{passive_pct}%</span>
                    </div>
                    <div style="width: 100%; background: #e9ecef; border-radius: 4px; height: 8px;">
                        <div style="width: {passive_pct}%; background: #FFA726; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <span style="color: #C62828; font-weight: 600;">😞 Detractors</span>
                        <span style="color: #2c3e50; font-weight: 700;">{sentiment_data["detractor_pct"]}%</span>
                    </div>
                    <div style="width: 100%; background: #e9ecef; border-radius: 4px; height: 8px;">
                        <div style="width: {sentiment_data["detractor_pct"]}%; background: #FF5252; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Account Health Overview
    st.subheader("🏢 Account Portfolio Health")

    if not portfolio.empty:
        col1, col2 = st.columns([3, 2])

        with col1:
            # Health distribution chart - using PERCENTAGES for better interpretability
            health_dist = portfolio["Status"].value_counts().reset_index()
            health_dist.columns = ["Status", "Count"]
            total_accounts = health_dist["Count"].sum()
            health_dist["Percentage"] = (
                health_dist["Count"] / total_accounts * 100
            ).round(1)

            color_map = {
                "Healthy": "#66BB6A",
                "Needs Attention": "#FFA726",
                "At Risk": "#FF5252",
            }

            fig = px.bar(
                health_dist,
                x="Status",
                y="Percentage",
                color="Status",
                color_discrete_map=color_map,
                template="plotly_white",
                text=health_dist.apply(
                    lambda row: f"{row['Percentage']}%<br>({row['Count']})", axis=1
                ),
            )

            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(255,255,255,0.9)",
                plot_bgcolor="rgba(248,249,250,1)",
                xaxis_title="",
                yaxis_title="Percentage of Portfolio",
                yaxis_range=[0, 100],
                font=dict(family="Inter", color="#2c3e50"),
            )

            fig.update_traces(textposition="outside")

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Priority Actions")

            at_risk = len(portfolio[portfolio["Status"] == "At Risk"])
            needs_attention = len(portfolio[portfolio["Status"] == "Needs Attention"])

            if at_risk > 0:
                at_risk_pct = round(at_risk / total_accounts * 100, 1)
                st.markdown(
                    f"""
                    <div class="alert-critical">
                        ⚠️ <strong>{at_risk} account(s)</strong> ({at_risk_pct}%) at risk of churn
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if needs_attention > 0:
                attention_pct = round(needs_attention / total_accounts * 100, 1)
                st.markdown(
                    f"""
                    <div class="alert-warning">
                        ⚡ <strong>{needs_attention} account(s)</strong> ({attention_pct}%) need proactive engagement
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            healthy = len(portfolio[portfolio["Status"] == "Healthy"])
            if healthy > 0:
                healthy_pct = round(healthy / total_accounts * 100, 1)
                st.markdown(
                    f"""
                    <div class="alert-success">
                        ✅ <strong>{healthy} account(s)</strong> ({healthy_pct}%) performing well
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "📊 Go to Action Center",
                on_click=navigate_to_page,
                args=("⚡ Action Center",),
                use_container_width=True,
            )

    st.markdown("---")

    # Word Cloud Section - Executive Summary
    st.subheader("☁️ Customer Voice Word Cloud")

    word_cloud_data = generate_word_clouds(real_df)

    if word_cloud_data["clouds"]:
        if word_cloud_data["has_time"]:
            st.markdown("*Word frequency visualization by quarter — hover for details*")

            # Create tabs for each time period
            if len(word_cloud_data["clouds"]) > 1:
                period_labels = [
                    f"📅 {period}" for period, _ in word_cloud_data["clouds"]
                ]
                tabs = st.tabs(period_labels)

                for i, (period, img_bytes) in enumerate(word_cloud_data["clouds"]):
                    with tabs[i]:
                        st.image(img_bytes, use_container_width=True)
            else:
                period, img_bytes = word_cloud_data["clouds"][0]
                st.image(img_bytes, use_container_width=True)
        else:
            st.markdown("*Aggregate word frequency from all customer feedback*")
            _, img_bytes = word_cloud_data["clouds"][0]
            st.image(img_bytes, use_container_width=True)
    else:
        st.info(
            "Word cloud generation requires text feedback data. Ensure the 'Interaction_Payload' column is populated."
        )
