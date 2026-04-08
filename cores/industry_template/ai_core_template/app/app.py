import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Core Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling & Custom CSS ---
st.markdown(
    """
<style>
    /* Global Styles */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    .stMetricLabel {
        color: #B0BEC5 !important;
    }
    .stMetricValue {
        color: #00E5FF !important;
        font-weight: 700 !important;
    }
    
    /* Card-like containers for metrics */
    div[data-testid="metric-container"] {
        background-color: #1E232E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2B3240;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #2B3240;
    }
    
    /* Charts background */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    
    /* Custom button aesthetics */
    .stButton > button {
        background: linear-gradient(90deg, #00C853 0%, #00E5FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Helper Functions for Fake Data ---
@st.cache_data
def load_data():
    # Simulate Model Metrics over time
    dates = pd.date_range(start="2024-01-01", periods=100)
    metrics_df = pd.DataFrame(
        {
            "Date": dates,
            "Accuracy": np.random.uniform(0.85, 0.98, 100),
            "Loss": np.random.uniform(0.1, 0.3, 100) * np.linspace(1, 0.5, 100),
            "Inference_Time_ms": np.random.normal(45, 5, 100),
        }
    )

    # Simulate recent predictions
    categories = ["Fraud", "Transaction", "Login", "Update"]
    recent_preds = pd.DataFrame(
        {
            "ID": [f"TRX-{1000 + i}" for i in range(20)],
            "Timestamp": [
                datetime.now() - timedelta(minutes=i * 15) for i in range(20)
            ],
            "Type": np.random.choice(categories, 20),
            "Confidence": np.random.uniform(0.7, 0.99, 20),
            "Status": np.random.choice(["Processed", "Flagged"], 20, p=[0.9, 0.1]),
        }
    )
    return metrics_df, recent_preds


metrics_df, recent_preds = load_data()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.title("AI Core")
    st.caption("v2.4.0-stable | 🟢 Online")

    st.markdown("---")

    menu = st.radio(
        "Navigation",
        ["Dashboard", "Model Performance", "Data Explorer", "Live Inference"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### System Health")
    st.progress(88, text="Memory Usage")
    st.caption("GPU Load: 42%")

# --- Main Content ---

if menu == "Dashboard":
    st.title("🚀 Executive Overview")
    st.markdown("Real-time monitoring of AI subsystems and model health.")

    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", "1.2M", "+12%")
    with col2:
        st.metric("Avg Latency", "45ms", "-5%")
    with col3:
        st.metric("Model Accuracy", "94.2%", "+0.8%")
    with col4:
        st.metric("Active Models", "5", "Stable")

    st.markdown("---")

    # Main Chart Row
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Traffic Volume & Anomalies")
        # Generate some nice curves
        fig_volume = px.area(
            metrics_df,
            x="Date",
            y="Inference_Time_ms",
            color_discrete_sequence=["#00E5FF"],
            template="plotly_dark",
        )
        fig_volume.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    with c2:
        st.subheader("Request Distribution")
        dist_data = recent_preds["Type"].value_counts().reset_index()
        dist_data.columns = ["Type", "Count"]
        fig_pie = px.doughnut(
            dist_data,
            values="Count",
            names="Type",
            hole=0.6,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            showlegend=False,
        )
        fig_pie.add_annotation(
            text="Distribution", showarrow=False, font=dict(color="white")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif menu == "Model Performance":
    st.title("📈 Model Analytics")

    tab1, tab2 = st.tabs(["Training Metrics", "Drift Analysis"])

    with tab1:
        st.subheader("Loss & Accuracy Trends")

        # Dual axis chart
        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=metrics_df["Date"],
                y=metrics_df["Accuracy"],
                name="Accuracy",
                line=dict(color="#00E5FF", width=3),
            )
        )
        fig_perf.add_trace(
            go.Scatter(
                x=metrics_df["Date"],
                y=metrics_df["Loss"],
                name="Loss",
                line=dict(color="#FF4081", width=3, dash="dot"),
                yaxis="y2",
            )
        )

        fig_perf.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Accuracy", range=[0, 1]),
            yaxis2=dict(title="Loss", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with tab2:
        st.info(
            "Drift analysis compares current distribution against baseline training data."
        )
        st.subheader("Feature Drift Heatmap")

        # Fake heatmap
        cols = ["Age", "Income", "Tenure", "Balance", "NumOfProducts"]
        drift_matrix = np.random.rand(5, 5)
        fig_heat = px.imshow(
            drift_matrix,
            x=cols,
            y=cols,
            color_continuous_scale="Viridis",
            labels=dict(color="Drift Score"),
        )
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)

elif menu == "Data Explorer":
    st.title("💾 Data Ledger")

    st.dataframe(
        recent_preds,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.2f", min_value=0, max_value=1
            ),
            "Status": st.column_config.Column("Status"),
        },
        use_container_width=True,
        hide_index=True,
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "Download CSV",
            data=recent_preds.to_csv(),
            file_name="audit_log.csv",
            mime="text/csv",
        )

elif menu == "Live Inference":
    st.title("⚡ Live Prediction Playground")

    with st.container():
        st.write("Test the model with custom inputs.")

        c_in1, c_in2 = st.columns(2)
        with c_in1:
            input_text = st.text_area(
                "Input Text / JSON payload",
                height=150,
                value='{"transaction_id": "12345", "amount": 500.00}',
            )

        with c_in2:
            st.markdown("#### Configuration")
            model_select = st.selectbox(
                "Select Model Version",
                ["v2.4.0 (Prod)", "v2.5.0-rc1 (Staging)", "v1.0.0 (Legacy)"],
            )
            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.75)

        if st.button("Run Prediction"):
            with st.spinner("Processing..."):
                # Simulate processing
                import time

                time.sleep(1)

                prediction_score = np.random.uniform(0.8, 0.99)
                is_flagged = prediction_score > threshold

                st.success("Inference Complete")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(
                        "Predicted Class",
                        "Legitimate" if not is_flagged else "Fraudulent",
                    )
                with res_col2:
                    st.metric("Confidence Score", f"{prediction_score:.4f}")

            st.json(
                {
                    "model": model_select,
                    "timestamp": datetime.now().isoformat(),
                    "result": "OK",
                    "latency_ms": 42,
                }
            )
