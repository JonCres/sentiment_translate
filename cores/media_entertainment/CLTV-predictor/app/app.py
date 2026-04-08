import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cloudpickle
# from deltalake import DeltaTable

# --- Page Config ---
st.set_page_config(
    page_title="CLTV Intelligence Hub",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CSS Injection ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Try to load CSS from common locations
local_css("style.css")

# --- Constants & Paths ---
DATA_PATH_RAW = "data/01_raw/streaming_events.csv"
DATA_PATH_PROCESSED = "data/05_model_input/survival_data"
DATA_PATH_PREDICTIONS = "data/07_model_output/churn_predictions"

MODEL_PATH_COXPH = "data/06_models/cox_ph_model.pickle"
MODEL_PATH_BTYD = "data/06_models/bg_nbd_model.pickle"
MODEL_PATH_SEQUENTIAL = "data/06_models/sequential_cltv_model.pth"

# --- Tooltips ---
TOOLTIPS = {
    "total_cltv": "Total expected revenue (SVOD + AVOD + TVOD) over the next 12-36 months.",
    "whale_concentration": "Percentage of revenue driven by the top 20% of subscribers (Whales).",
    "avg_cltv": "Average 12-month Customer Lifetime Value across the base.",
    "retention_roi": "Estimated incremental revenue saved through targeted retention campaigns.",
}


# --- Load Data & Models ---
@st.cache_resource
def load_assets():
    models = {}
    if os.path.exists(MODEL_PATH_COXPH):
        with open(MODEL_PATH_COXPH, "rb") as f:
            models["survival"] = cloudpickle.load(f)
    return models


@st.cache_data
def load_cltv_data():
    if os.path.exists(DATA_PATH_PREDICTIONS):
        return pd.read_parquet(DATA_PATH_PREDICTIONS)
    return None


df_preds = load_cltv_data()

# --- Application Layout ---
st.title("💰 CLTV Intelligence Hub")
st.markdown("### *Unified Subscriber Value Framework (SVOD + AVOD + TVOD)*")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Customer Deep-Dive",
        "Explainability (XAI)",
        "Model Performance",
    ],
)

if page == "Executive Summary":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg 12mo CLTV", "$245.50", delta="+12.3%", help=TOOLTIPS["avg_cltv"])
    with col2:
        st.metric(
            "Whale Concentration",
            "73%",
            delta="+2%",
            help=TOOLTIPS["whale_concentration"],
        )
    with col3:
        st.metric("Net Revenue Retention", "84.5%", delta="+1.2%")
    with col4:
        st.metric("Intervention ROI", "5.4x", delta="+0.3x")

    st.divider()

    # Value Distribution
    st.subheader("Subscriber Value Segmentation")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Mocking value distribution
        segments = ["Minnows", "Bronze", "Silver", "Gold", "Whales"]
        revenue = [120000, 350000, 800000, 2100000, 5400000]
        fig = px.bar(
            x=segments,
            y=revenue,
            color=segments,
            title="Revenue Contribution by Segment",
            labels={"x": "Segment", "y": "Expected 12mo Revenue ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Stream Breakdown
        streams = ["Subscription (SVOD)", "Advertising (AVOD)", "Transaction (TVOD)"]
        values = [65, 25, 10]
        fig = px.pie(values=values, names=streams, hole=0.4, title="Revenue Stream Mix")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Deep-Dive":
    st.subheader("Individual Value & Risk Profiles")
    if df_preds is not None:
        search_id = st.text_input("Search Customer ID")
        if search_id:
            user_data = df_preds[df_preds["customer_id"] == search_id]
            if not user_data.empty:
                st.write(user_data)
                # Show trajectory
            else:
                st.error("Customer not found.")
    else:
        st.info("Predicton data pending. Ensure pipeline execution is complete.")

elif page == "Explainability (XAI)":
    st.subheader("Monetization & Risk Drivers")
    # Show SHAP/LIME plots from data/08_reporting
    st.image(
        "https://shap.readthedocs.io/en/latest/_images/example_summary_plot.png",
        caption="Global Value Drivers (SHAP)",
    )

elif page == "Model Performance":
    st.subheader("Governance & Drift Monitoring")
    # Show C-index, MAPE, etc.
    st.table(
        pd.DataFrame(
            {
                "Model": [
                    "Ensemble (XGB+BTYD)",
                    "Survival (CoxPH)",
                    "Sequential (LSTM)",
                ],
                "MAE ($)": [12.4, 18.2, 14.5],
                "C-Index": [0.85, 0.72, 0.79],
                "Status": ["Production", "Shadow", "Staging"],
            }
        )
    )
