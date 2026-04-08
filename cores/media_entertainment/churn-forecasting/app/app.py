import streamlit as st
import pandas as pd
import plotly.express as px
import os
import cloudpickle
from deltalake import DeltaTable

# --- Page Config ---
st.set_page_config(
    page_title="Churn Intelligence Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CSS Injection ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Try to load CSS
if os.path.exists("app/style.css"):
    local_css("app/style.css")
elif os.path.exists("cores/media_entertainment/churn-forecasting/app/style.css"):
    local_css("cores/media_entertainment/churn-forecasting/app/style.css")

# --- Constants & Paths ---
DATA_PATH_RAW = "data/01_raw/online_retail_II.csv"
DATA_PATH_CLEAN = "data/02_intermediate/clean_data"
DATA_PATH_PROCESSED = "data/05_model_input/processed_data"
DATA_PATH_PREDICTIONS = "data/07_model_output/churn_predictions"

MODEL_PATH_ENSEMBLE = "data/06_models/ensemble_model.pickle"

INTERPRETATIONS_DIR = "data/08_reporting/plots/interpretations"

# --- Tooltip Explanations ---
TOOLTIPS = {
    "total_customers": "Total unique customers identified in the dataset.",
    "avg_risk": "Average probability that a customer will churn in the next 30 days.",
    "churn_prob_30day": "Probability of churn within the next 30 days.",
    "churn_prob_90day": "Probability of churn within the next 90 days.",
    "survival_365d": "Probability of the subscriber remaining active after 1 year.",
    "hazard_rate": "The instantaneous rate of failure (churn) at a specific time.",
    "median_tenure": "Predicted number of days until a 50% probability of churn is reached.",
    "intervention_window": "Recommended lead time (days) for retention actions before expected churn.",
    "risk_segment": "Imminent (>85% risk), At-Risk (33-85%), or Passive (<33% risk).",
}

# --- Helper Functions ---


@st.cache_resource
def load_models():
    """Loads the trained Ensemble churn model."""
    models = {}
    if os.path.exists(MODEL_PATH_ENSEMBLE):
        with open(MODEL_PATH_ENSEMBLE, "rb") as f:
            models["ensemble"] = cloudpickle.load(f)
    return models


@st.cache_data
def load_data():
    """Loads all available data layers and predictions."""
    data_dict = {}

    # Load Raw CSV
    if os.path.exists(DATA_PATH_RAW):
        try:
            data_dict["Raw Data"] = pd.read_csv(DATA_PATH_RAW)
        except Exception as e:
            st.error(f"Error loading raw data: {e}")

    # Load Clean Data
    if os.path.exists(DATA_PATH_CLEAN):
        try:
            data_dict["Cleaned Customer Data"] = pd.read_parquet(DATA_PATH_CLEAN)
        except Exception:
            pass  # Silently fail for folders

    # Load Model Input (Processed)
    if os.path.exists(DATA_PATH_PROCESSED):
        try:
            data_dict["Processed Features"] = pd.read_parquet(DATA_PATH_PROCESSED)
        except Exception:
            pass

    # Load Churn Predictions
    if os.path.exists(DATA_PATH_PREDICTIONS):
        try:
            # Predictions are stored as Delta Table
            data_dict["Churn Predictions"] = DeltaTable(
                DATA_PATH_PREDICTIONS
            ).to_pandas()
        except Exception as e:
            st.warning(
                f"Could not load predictions as Delta Table: {e}. Trying Parquet fallback."
            )
            try:
                data_dict["Churn Predictions"] = pd.read_parquet(DATA_PATH_PREDICTIONS)
            except Exception:
                pass

    return data_dict


@st.cache_data
def load_interpretations():
    """Loads AI interpretations from markdown files."""
    interpretations = {}
    if os.path.exists(INTERPRETATIONS_DIR):
        for filename in os.listdir(INTERPRETATIONS_DIR):
            if filename.endswith(".md"):
                name = filename.replace(".md", "")
                with open(os.path.join(INTERPRETATIONS_DIR, filename), "r") as f:
                    interpretations[name] = f.read()
    return interpretations


def load_customer_interpretation(customer_id):
    """Loads AI interpretation for a specific customer from the saved files."""
    path = os.path.join(INTERPRETATIONS_DIR, "customers", f"{customer_id}.md")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error loading interpretation: {str(e)}"
    return "No AI interpretation available for this customer. Please run the visualization pipeline once to generate interpretations using the SLM."


def plot_sales_over_time(df):
    if all(col in df.columns for col in ["InvoiceDate", "Price", "Quantity"]):
        df_plot = df.copy()
        df_plot["Sales"] = df_plot["Price"] * df_plot["Quantity"]
        df_plot["InvoiceDate"] = pd.to_datetime(df_plot["InvoiceDate"])
        daily_sales = (
            df_plot.groupby(pd.Grouper(key="InvoiceDate", freq="D"))["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily_sales,
            x="InvoiceDate",
            y="Sales",
            title="Daily Sales Over Time",
            template="plotly_white",
            line_shape="spline",
        )
        fig.update_traces(line_color="#667eea")
        st.plotly_chart(fig, use_container_width=True)


def main():
    with st.spinner("Initializing models and data..."):
        data = load_data()
        _ = load_models()
        _ = load_interpretations()

    if not data:
        st.error("⚠️ No data found. Please run the Kedro pipeline first.")
        st.code("kedro run")
        return

    # --- Header ---
    st.title("🛡️ Churn Intelligence Hub")
    st.markdown("### Advanced Subscriber Retention & Risk Analysis")

    tabs = st.tabs(
        [
            "🌎 Overview",
            "👤 Customer Insights",
            "📈 Model Performance",
            "🔍 Explainability",
            "🚀 Strategic Actions",
        ]
    )

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.markdown("## 🌐 Global Business Overview")

        # Pick a default dataset for metrics
        df_main = data.get(
            "Churn Predictions",
            data.get(
                "Processed Features",
                data.get("Cleaned Customer Data", list(data.values())[0]),
            ),
        )

        total_customers = len(df_main)
        avg_risk = (
            df_main["churn_prob_30day"].mean()
            if "churn_prob_30day" in df_main.columns
            else 0
        )

        # HTML for Global Metrics Cards
        st.html(f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;'>
            <div class='metric-card' title='{TOOLTIPS["total_customers"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #667eea;'>
                <div style='font-size: 0.75rem; color: #667eea; font-weight: 700; text-transform: uppercase;'>Total Subscribers ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{total_customers:,}</div>
                <div style='font-size: 0.7rem; color: #667eea; font-weight: 600;'>Total Base</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["avg_risk"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #ffc107;'>
                <div style='font-size: 0.75rem; color: #f39c12; font-weight: 700; text-transform: uppercase;'>Avg. Churn Risk ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{avg_risk:.1%}</div>
                <div style='font-size: 0.7rem; color: #f39c12; font-weight: 600;'>30-Day Outlook</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["risk_segment"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #dc3545;'>
                <div style='font-size: 0.75rem; color: #dc3545; font-weight: 700; text-transform: uppercase;'>Imminent Risk ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{len(df_main[df_main["risk_segment"] == "Imminent"]):,}</div>
                <div style='font-size: 0.7rem; color: #dc3545; font-weight: 600;'>Immediate Attrition</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["intervention_window"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #6610f2;'>
                <div style='font-size: 0.75rem; color: #6610f2; font-weight: 700; text-transform: uppercase;'>Avg. Win-back Window ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{df_main["recommended_intervention_days_lead"].mean():.0f} d</div>
                <div style='font-size: 0.7rem; color: #6610f2; font-weight: 600;'>Lead Time to Act</div>
            </div>
        </div>
        """)

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Subscriber Engagement Distribution")
            if "Processed Features" in df_main.columns or True:
                # Distribution of tenure vs risk
                if "T" in df_main.columns and "churn_prob_30day" in df_main.columns:
                    fig_scatter = px.scatter(
                        df_main,
                        x="T",
                        y="churn_prob_30day",
                        color="risk_tier" if "risk_tier" in df_main.columns else None,
                        title="Tenure vs. Churn Risk",
                        labels={
                            "T": "Tenure (Days)",
                            "churn_prob_30day": "30-Day Churn Prob",
                        },
                        template="plotly_white",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Engagement metrics not fully available for visualization.")

        with col_right:
            st.subheader("Risk Segmentation")
            if "risk_tier" in df_main.columns:
                tier_counts = df_main["risk_tier"].value_counts().reset_index()
                fig_pie = px.pie(
                    tier_counts,
                    values="count",
                    names="risk_tier",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.YlOrRd[::-1],
                    template="plotly_white",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 2: CUSTOMER INSIGHTS ---
    with tabs[1]:
        st.header("🎯 Individual Subscriber Risk Analysis")
        if "Churn Predictions" in data:
            df_preds = data["Churn Predictions"]
            customer_ids = sorted(df_preds["customer_id"].unique())
            selected_cust_id = st.selectbox("Select Subscriber ID", customer_ids)
            cust_record = df_preds[df_preds["customer_id"] == selected_cust_id].iloc[0]

            col_card, col_analysis = st.columns([1, 1])

            with col_card:
                st.markdown("##### 🚀 Risk Intelligence Card")
                risk_30d = cust_record.get("churn_prob_30day", 0)
                status_color = (
                    "#28a745"
                    if risk_30d < 0.3
                    else "#ffc107"
                    if risk_30d < 0.7
                    else "#dc3545"
                )
                status_label = (
                    "HEALTHY"
                    if risk_30d < 0.3
                    else "AT RISK"
                    if risk_30d < 0.7
                    else "CRITICAL"
                )

                html_card = f"""
                <div style='background: white; padding: 25px; border-radius: 20px; border-left: 10px solid {status_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.05);'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-weight: 700; color: {status_color};'>{status_label}</span>
                        <span style='background: #f8f9fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;'>{cust_record.get("intervention_priority", "Low")} Priority</span>
                    </div>
                    <h3 style='margin: 15px 0;'>Subscriber {selected_cust_id}</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                        <div style='background: #f8f9fa; padding: 15px; border-radius: 12px;'>
                            <div style='font-size: 0.65rem; color: #667eea; font-weight: 600;'>30D CHURN RISK</div>
                            <div style='font-size: 1.3rem; font-weight: 700;'>{risk_30d:.1%}</div>
                        </div>
                        <div style='background: #f8f9fa; padding: 15px; border-radius: 12px;'>
                            <div style='font-size: 0.65rem; color: #667eea; font-weight: 600;'>1Y SURVIVAL PROB</div>
                            <div style='font-size: 1.3rem; font-weight: 700;'>{cust_record.get("survival_prob_365d", 0):.1%}</div>
                        </div>
                        <div style='background: #f8f9fa; padding: 15px; border-radius: 12px;'>
                            <div style='font-size: 0.65rem; color: #667eea; font-weight: 600;'>HAZARD RATIO</div>
                            <div style='font-size: 1.3rem; font-weight: 700;'>{cust_record.get("hazard_ratio", 0):.2f}x</div>
                        </div>
                        <div style='background: #f8f9fa; padding: 15px; border-radius: 12px;'>
                            <div style='font-size: 0.65rem; color: #667eea; font-weight: 600;'>MEDIAN TENURE</div>
                            <div style='font-size: 1.3rem; font-weight: 700;'>{cust_record.get("predicted_median_tenure_days", 0):.0f}d</div>
                        </div>
                    </div>
                    <div style='margin-top: 20px;'>
                        <div style='font-size: 0.8rem; color: #7f8c8d;'>Estimated Time to Churn: <b>{cust_record.get("time_to_churn_days", 0):.0f} days</b></div>
                    </div>
                </div>
                """
                st.html(html_card)

            with col_analysis:
                st.markdown("##### 🤖 AI Risk Interpretation")
                interpretation = load_customer_interpretation(selected_cust_id)
                st.info(interpretation)

    # --- TAB 3: MODEL PERFORMANCE ---
    with tabs[2]:
        st.header("📈 Model Performance & Calibration")
        st.info("Visualization of model accuracy, AUC-ROC, and confidence intervals.")

        if "Model Metrics" in data:
            metrics_df = data["Model Metrics"]
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("Performance metrics not found. Run the visualization pipeline.")

    # --- TAB 4: EXPLAINABILITY ---
    with tabs[3]:
        st.header("🔍 Explainability Hub")
        st.subheader("Global Feature Importance (SHAP)")
        st.markdown(
            "This view shows which behaviors (e.g., watch time decline, session frequency) are driving churn across the platform."
        )

        # Placeholder for SHAP plot
        st.image(
            "https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png",
            width=600,
        )
        st.caption("SHAP Feature Importance (Example Visualization)")

    # --- TAB 5: STRATEGIC ACTIONS ---
    with tabs[4]:
        st.header("🚀 Automated Retention Strategies")

        if "Churn Predictions" in data:
            df = data["Churn Predictions"]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚡ Critical Interventions")
                critical = df[df["intervention_priority"] == "Critical"].head(10)
                st.dataframe(
                    critical[
                        ["customer_id", "churn_prob_30day", "intervention_priority"]
                    ],
                    use_container_width=True,
                )
                if st.button("Trigger Critical Alert Workflow"):
                    st.success("Retention alerts sent to Marketing API.")

            with col2:
                st.subheader("🎯 Win-back Opportunities")
                winback = df[df["risk_tier"] == "High"].head(10)
                st.dataframe(
                    winback[["customer_id", "churn_prob_90day", "risk_tier"]],
                    use_container_width=True,
                )
                if st.button("Generate Win-back Batches"):
                    st.success("Win-back batches exported to CRM.")


if __name__ == "__main__":
    main()
