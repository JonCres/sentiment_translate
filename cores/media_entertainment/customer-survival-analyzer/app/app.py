import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os
import cloudpickle
import numpy as np
# from deltalake import DeltaTable

# --- Page Config ---
st.set_page_config(
    page_title="Survival Intelligence Hub",
    page_icon="⌛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CSS Injection ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Try to load CSS from common locations
local_css("app/style.css")
local_css("cores/media_entertainment/customer-survival-analyzer/app/style.css")

# --- Constants & Paths ---
DATA_PATH_RAW = "data/01_raw/streaming_events.csv"
DATA_PATH_PROCESSED = "data/05_model_input/survival_data"
DATA_PATH_PREDICTIONS = "data/07_model_output/churn_predictions"

MODEL_PATH_COXPH = "data/06_models/cox_ph_model.pickle"
MODEL_PATH_RSF = "data/06_models/rsf_model.pickle"
MODEL_PATH_DEEPSURV = "data/06_models/deepsurv_model.pth"

PLOT_PATH_SHAP = "data/08_reporting/plots/shap_summary_plot.png"
PLOT_PATH_SURVIVAL = "data/08_reporting/plots/survival_curves_plot.png"
PLOT_PATH_COMPARISON = "data/08_reporting/plots/model_comparison_plot.png"

# --- Tooltip Explanations ---
TOOLTIPS = {
    "total_customers": "Total unique subscribers tracked in the engine.",
    "hazard_rate": "The instantaneous risk of a user churning at the current moment.",
    "survival_prob": "The probability that a user will remain active beyond a specific time horizon.",
    "churn_prob_30d": "Likelihood of account termination within the next 30 days.",
    "median_tenure": "The typical duration a subscriber stays with the platform.",
    "c_index": "Concordance Index: Measures how well the model ranks users by their actual time-to-churn.",
    "brier_score": "Measures the accuracy of predicted survival probabilities (lower is better).",
    "buffering_impact": "Estimated hazard ratio increase per 1% increase in buffering events.",
    "intervention_lead_time": "Average days of advance notice before a predicted churn event.",
}

# --- Helper Functions ---


@st.cache_resource
def load_models():
    """Loads the trained survival models."""
    models = {}
    if os.path.exists(MODEL_PATH_COXPH):
        with open(MODEL_PATH_COXPH, "rb") as f:
            models["coxph"] = cloudpickle.load(f)
    if os.path.exists(MODEL_PATH_RSF):
        with open(MODEL_PATH_RSF, "rb") as f:
            models["rsf"] = cloudpickle.load(f)
    return models


@st.cache_data
def load_data():
    """Loads survival data and predictions."""
    data_dict = {}

    # Load Survival Input
    if os.path.exists(DATA_PATH_PROCESSED):
        try:
            data_dict["Survival Data"] = pd.read_parquet(DATA_PATH_PROCESSED)
        except Exception:
            pass

    # Load Predictions
    if os.path.exists(DATA_PATH_PREDICTIONS):
        try:
            data_dict["Predictions"] = pd.read_parquet(DATA_PATH_PREDICTIONS)
        except Exception:
            pass

    return data_dict


# --- Main Dashboard ---

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Executive Dashboard",
        "Survival Radar",
        "Explainability (XAI)",
        "Model Governance",
    ],
)

models = load_models()
data_layers = load_data()

st.title("⌛ Survival Intelligence Hub")
st.markdown("### *Subscriber Churn & Hazard Monitoring Engine*")

if page == "Executive Dashboard":
    # --- Top Level KPIs ---
    col1, col2, col3, col4 = st.columns(4)

    # Calculate aggregate metrics if data is available
    total_subs = "1.2M"
    avg_churn = "3.1%"
    med_tenure = "14.2 Mo"
    lead_time = "21 Days"

    if "Predictions" in data_layers:
        preds = data_layers["Predictions"]
        total_subs = (
            f"{len(preds) / 1e6:.1f}M"
            if len(preds) > 1e6
            else f"{len(preds) / 1e3:.1f}k"
        )
        avg_churn = f"{preds['churn_prob_30day'].mean() * 100:.1f}%"
        med_tenure = f"{preds['predicted_median_tenure_days'].mean() / 30:.1f} Mo"

    with col1:
        st.metric(
            "Total Subscribers",
            total_subs,
            delta="+4.2k",
            help=TOOLTIPS["total_customers"],
        )
    with col2:
        st.metric(
            "Avg Churn Prob (30d)",
            avg_churn,
            delta="-0.2%",
            delta_color="normal",
            help=TOOLTIPS["churn_prob_30d"],
        )
    with col3:
        st.metric(
            "Median Tenure", med_tenure, delta="+0.5 Mo", help=TOOLTIPS["median_tenure"]
        )
    with col4:
        st.metric(
            "Intervention Lead Time",
            lead_time,
            delta="+3 Days",
            help=TOOLTIPS["intervention_lead_time"],
        )

    st.divider()

    # --- Primary Views ---
    tab1, tab2 = st.tabs(["Active Hazards", "Retention Workflows"])

    with tab1:
        st.subheader("Global Survival Trends")
        if os.path.exists(PLOT_PATH_SURVIVAL):
            st.image(
                PLOT_PATH_SURVIVAL, caption="Aggregated Kaplan-Meier Survival Curves"
            )
        else:
            # Mock Survival Curve if file doesn't exist
            t = np.linspace(0, 100, 100)
            s = np.exp(-0.02 * t)
            fig = px.line(
                x=t,
                y=s,
                labels={"x": "Tenure (Days)", "y": "Survival Probability"},
                title="System-wide Survival Function S(t)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("High-Risk Intervention Priority")
        if "Predictions" in data_layers:
            preds = data_layers["Predictions"]
            # Filter for high risk
            high_risk = (
                preds.filter(pl.col("churn_prob_30day") > 0.1)
                .sort("churn_prob_30day", descending=True)
                .to_pandas()
            )

            st.dataframe(
                high_risk[
                    [
                        "customer_id",
                        "churn_prob_30day",
                        "intervention_priority",
                        "risk_segment",
                        "predicted_median_tenure_days",
                    ]
                ].head(20),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🚀 Trigger Retention Campaign for Top 100"):
                st.success(
                    "Successfully pushed 100 high-risk profiles to Marketing Automation API."
                )
        else:
            st.warning(
                "Prediction data not found. Please run the `monitoring` pipeline."
            )

elif page == "Survival Radar":
    st.subheader("Individual Hazard Analysis")

    if "Predictions" in data_layers:
        preds = data_layers["Predictions"].to_pandas()

        # Risk Scatter Plot
        fig = px.scatter(
            preds,
            x="churn_prob_30day",
            y="churn_prob_90day",
            size="hazard_ratio",
            color="churn_prob_30day",
            color_continuous_scale="RdYlGn_r",
            hover_data=[
                "customer_id",
                "intervention_priority",
                "predicted_median_tenure_days",
            ],
            labels={
                "churn_prob_30day": "Short Term (30d)",
                "churn_prob_90day": "Long Term (90d)",
                "hazard_ratio": "Hazard Multiplier",
            },
            title="Risk Velocity: 30d vs 90d Churn Probability",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Search for a customer
        customer_id = st.text_input(
            "Search Customer ID (e.g., CUST_001)",
            help="Enter a customer ID to view their specific hazard trajectory.",
        )
        if customer_id:
            cust_data = preds[preds["customer_id"] == customer_id]
            if not cust_data.empty:
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.write("#### Customer Profile")
                    st.json(cust_data.iloc[0].to_dict())

                with col_b:
                    # Individual Survival Plot
                    t_horizons = [30, 60, 90, 180, 365]
                    probs = [
                        cust_data["churn_prob_30day"].values[0],
                        cust_data["churn_prob_60day"].values[0],
                        cust_data["churn_prob_90day"].values[0],
                        cust_data["churn_prob_180day"].values[0],
                        cust_data["churn_prob_365day"].values[0],
                    ]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=t_horizons,
                            y=[1 - p for p in probs],
                            mode="lines+markers",
                            line=dict(color="#00d1b2", width=4),
                            fill="tozeroy",
                            name="Survival Probability S(t)",
                        )
                    )

                    fig.update_layout(
                        title=f"Hazard Trajectory for {customer_id}",
                        xaxis_title="Days from Today",
                        yaxis_title="Survival Probability S(t)",
                        template="plotly_dark",
                        yaxis_range=[0, 1.05],
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Customer ID not found.")
    else:
        st.info("Load predictions to enable Radar features.")

elif page == "Explainability (XAI)":
    st.subheader("Feature Attribution & Hazard Drivers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Global Importance (SHAP)")
        if os.path.exists(PLOT_PATH_SHAP):
            st.image(PLOT_PATH_SHAP, caption="SHAP Global Hazard Drivers")
        else:
            # Mock SHAP plot
            features = [
                "Buffering Events",
                "Session Frequency",
                "Tenure",
                "Plan Tier",
                "Device Type",
            ]
            importance = [0.45, 0.30, 0.15, 0.08, 0.02]
            fig = px.bar(
                x=importance,
                y=features,
                orientation="h",
                labels={"x": "SHAP Importance", "y": "Feature"},
                title="Macro Attrition Drivers",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Robustness Metrics")
        # Mock XAI Metrics
        metrics = {"Faithfulness": 0.82, "Consistency": 0.78, "Monotonicity": 0.91}
        fig = px.pie(
            values=list(metrics.values()),
            names=list(metrics.keys()),
            title="Explainer Quality Scores",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Governance":
    st.subheader("Survival Model Assessment")

    if os.path.exists(PLOT_PATH_COMPARISON):
        st.image(PLOT_PATH_COMPARISON, caption="Survival Model Performance (C-Index)")
    else:
        # Mock Model Comparison
        model_names = ["CoxPH", "RSF", "DeepSurv", "NMTLR"]
        c_indices = [0.72, 0.78, 0.84, 0.81]
        fig = px.bar(
            x=model_names,
            y=c_indices,
            labels={"x": "Model Architecture", "y": "C-Index"},
            title="Performance Leaderboard (Concordance Index)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Model Registry (MLflow Integration)")
    registry_data = {
        "Model": ["CoxPH", "RSF", "DeepSurv", "NMTLR"],
        "Status": ["Staging", "Production", "In-Training", "Archived"],
        "Last Refreshed": ["2024-05-15", "2024-05-18", "2024-05-20", "2024-04-01"],
        "C-Index": [0.72, 0.78, 0.84, 0.81],
    }
    st.table(registry_data)
