import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import cloudpickle
from lifetimes import BetaGeoFitter, GammaGammaFitter
from deltalake import DeltaTable

# --- Page Config ---
st.set_page_config(
    page_title="CLTV Insights Dashboard",
    page_icon="💎",
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
elif os.path.exists("cores/media_entertainment/predictive-CLTV-insights/app/style.css"):
    local_css("cores/media_entertainment/predictive-CLTV-insights/app/style.css")

# --- Constants & Paths ---
DATA_PATH_RAW = "data/01_raw/online_retail_II.csv"
DATA_PATH_CLEAN = "data/02_intermediate/clean_data"
DATA_PATH_PROCESSED = "data/05_model_input/processed_data"
DATA_PATH_PREDICTIONS = "data/07_model_output/cltv_predictions"

MODEL_PATH_BGNBD = "data/06_models/bg_nbd_model.pickle"
MODEL_PATH_GG = "data/06_models/gamma_gamma_model.pickle"

INTERPRETATIONS_DIR = "data/08_reporting/plots/interpretations"

# --- Tooltip Explanations ---
TOOLTIPS = {
    "total_customers": "Total unique customers identified in the dataset.",
    "avg_ticket": "Average monetary value generated per transaction.",
    "avg_freq": "Average number of repeat transactions per customer.",
    "predicted_cltv": "Total projected revenue from the current customer base over the next 12 months.",
    "churn_risk": "The probability that a customer has already churned or will become inactive in the next 30 days.",
    "p_alive": "The probability that the customer is currently 'active' or 'alive'.",
    "clv_12mo": "Projected net profit attributed to the entire future relationship with a customer for the next 12 months.",
    "clv_24mo": "Projected value over a 24-month horizon.",
    "clv_36mo": "Projected value over a 3-year horizon.",
    "expected_tx": "Predicted number of purchases the customer is expected to make in the coming year.",
    "avg_tx_val": "Predicted average revenue per future transaction for this specific customer.",
    "model_conf": "Statistical confidence of the model's prediction based on historical data consistency.",
    "ci": "The statistical range (95% certainty) where the actual future value is likely to fall.",
    "churn_impact": "Estimated revenue saved by preventing 1% of predicted churn through proactive retention.",
    "elite_count": "Number of customers belonging to the top 20% (Elite) segment by predicted value."
}

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """Loads the trained BG/NBD and Gamma-Gamma models."""
    models = {}
    if os.path.exists(MODEL_PATH_BGNBD):
        with open(MODEL_PATH_BGNBD, "rb") as f:
            models["bg_nbd"] = cloudpickle.load(f)
    if os.path.exists(MODEL_PATH_GG):
        with open(MODEL_PATH_GG, "rb") as f:
            models["gamma_gamma"] = cloudpickle.load(f)
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
            pass # Silently fail for folders

    # Load Model Input (Processed)
    if os.path.exists(DATA_PATH_PROCESSED):
        try:
            data_dict["Processed Features"] = pd.read_parquet(DATA_PATH_PROCESSED)
        except Exception:
            pass

    # Load CLTV Predictions
    if os.path.exists(DATA_PATH_PREDICTIONS):
        try:
            # Predictions are stored as Delta Table
            data_dict["CLTV Predictions"] = DeltaTable(DATA_PATH_PREDICTIONS).to_pandas()
        except Exception as e:
            st.warning(f"Could not load predictions as Delta Table: {e}. Trying Parquet fallback.")
            try:
                data_dict["CLTV Predictions"] = pd.read_parquet(DATA_PATH_PREDICTIONS)
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
        daily_sales = df_plot.groupby(pd.Grouper(key="InvoiceDate", freq="D"))["Sales"].sum().reset_index()

        fig = px.line(daily_sales, x="InvoiceDate", y="Sales", 
                      title="Daily Sales Over Time",
                      template="plotly_white",
                      line_shape="spline")
        fig.update_traces(line_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)

def main():
    
    with st.spinner("Initializing models and data..."):
        data = load_data()
        models = load_models()
        interpretations = load_interpretations()

    if not data:
        st.error("⚠️ No data found. Please run the Kedro pipeline first.")
        st.code("kedro run")
        return

    # --- Header ---
    st.title("📊 Predictive CLTV Insights")
    st.markdown("### Advanced Customer Lifetime Value Analysis & Dashboard")
    
    tabs = st.tabs(["🌎 Overview", "👤 Customer Insights", "🧬 Model Explorer", "📈 Model Insights", "🚀 Strategic KPIs"])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.markdown("## 🌐 Global Business Overview")
        
        # Pick a default dataset for metrics, prioritizing predictions if available
        df_main = data.get("CLTV Predictions", data.get("Processed Features", data.get("Cleaned Customer Data", list(data.values())[0])))
        
        # Calculate values
        total_customers = len(df_main)
        avg_ticket = df_main['monetary_value'].mean() if "monetary_value" in df_main.columns else 0
        avg_freq = df_main['frequency'].mean() if "frequency" in df_main.columns else 0
        total_cltv = df_main['clv_12mo'].sum() if "clv_12mo" in df_main.columns else 0
        
        # HTML for Global Metrics Cards with Title Tooltips
        st.html(f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;'>
            <div class='metric-card' title='{TOOLTIPS["total_customers"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #667eea; transition: all 0.3s ease; cursor: help;'>
                <div style='font-size: 0.75rem; color: #667eea; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>Total Customers ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{total_customers:,}</div>
                <div style='font-size: 0.7rem; color: #28a745; font-weight: 600;'>Active Base</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["avg_ticket"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #764ba2; transition: all 0.3s ease; cursor: help;'>
                <div style='font-size: 0.75rem; color: #764ba2; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>Avg. Ticket ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>${avg_ticket:.2f}</div>
                <div style='font-size: 0.7rem; color: #667eea; font-weight: 600;'>Per Transaction</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["avg_freq"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #28a745; transition: all 0.3s ease; cursor: help;'>
                <div style='font-size: 0.75rem; color: #28a745; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>Avg. Frequency ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{avg_freq:.2f}</div>
                <div style='font-size: 0.7rem; color: #28a745; font-weight: 600;'>Annual Orders</div>
            </div>
            <div class='metric-card' title='{TOOLTIPS["predicted_cltv"]}' style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 5px solid #ffc107; transition: all 0.3s ease; cursor: help;'>
                <div style='font-size: 0.75rem; color: #f39c12; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>Predicted CLTV ⓘ</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin: 10px 0;'>${total_cltv:,.0f}</div>
                <div style='font-size: 0.7rem; color: #667eea; font-weight: 600;'>12 Month Forecast</div>
            </div>
        </div>
        """)

        st.markdown("---")
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Sales Trend & Customer Behavior")
            if "Raw Data" in data:
                plot_sales_over_time(data["Raw Data"])
            else:
                st.info("Raw data not available for sales trend.")
        
        with col_right:
            st.subheader("Dataset Selection")
            choice = st.selectbox("Select layer to explore", list(data.keys()))
            st.dataframe(data[choice].head(20), use_container_width=True)
            
        st.markdown("---")
        
        st.header("🚨 Customers to pay attention to")
        if "CLTV Predictions" in data:
            df = data["CLTV Predictions"]
            
            # Helper to create priority lists
            def get_priority_list(condition, cols=["customer_id", "clv_12mo", "p_churn_30day", "clv_segment"]):
                return df[condition][cols].sort_values("clv_12mo", ascending=False)

            col_act, col_camp = st.columns(2)

            with col_act:
                st.subheader("⚡ Triggered Actions")
                
                # 1. Proactive Retention Workflow (High Churn Risk)
                retention_list = get_priority_list(df["p_churn_30day"] > 0.8)
                with st.expander(f"📉 Proactive Retention ({len(retention_list)})", expanded=False):
                    st.caption("Auto-triggering for users with Churn Risk > 80% before renewal.")
                    st.dataframe(retention_list, use_container_width=True, hide_index=True)
                
                # 2. High-Value Support Routing (Q5 Elite with Risk)
                support_list = get_priority_list((df["clv_segment"] == "Q5_Elite") & (df["p_churn_30day"] > 0.4))
                with st.expander(f"💎 High-Value Support Routing ({len(support_list)})"):
                    st.caption("Elite (Q5) customers flagged for VIP concierge due to detected risk.")
                    st.dataframe(support_list, use_container_width=True, hide_index=True)
                
                # 3. Bridge Content Injection (Proxy: Healthy customers with dropping frequency)
                # If we don't have 'frequency_delta', we use a proxy
                bridge_list = get_priority_list((df["p_alive"] > 0.5) & (df["expected_transactions_12mo"] < df["frequency"].mean()))
                with st.expander(f"📺 Bridge Content Injection ({len(bridge_list)})"):
                    st.caption("Content exhaustion detected. Triggering Recommendation Engine for habit-forming content.")
                    st.dataframe(bridge_list, use_container_width=True, hide_index=True)

            with col_camp:
                st.subheader("🎯 Targeted Campaigns")
                
                # 1. Monthly-to-Annual Upsell
                upsell_list = get_priority_list((df["p_alive"] > 0.9) & (df["expected_transactions_12mo"] > 5))
                with st.expander(f"🔄 Monthly-to-Annual Upsell ({len(upsell_list)})", expanded=False):
                    st.caption("Stable high-value users. Offer 50% off when switching to Annual.")
                    st.dataframe(upsell_list, use_container_width=True, hide_index=True)

                # 2. Whale Win-Back
                winback_list = get_priority_list((df["clv_segment"].isin(["Q5_Elite", "Q4_High"])) & (df["p_alive"] < 0.2))
                with st.expander(f"🐋 Whale Win-Back ({len(winback_list)})"):
                    st.caption("Former high-value segments (inactive > 90 days). Deep discount bundles.")
                    st.dataframe(winback_list, use_container_width=True, hide_index=True)
                
                # 3. Binge-and-Burnout Prevention
                # Proxy: High frequency but high p_churn_90day
                burnout_list = get_priority_list((df["frequency"] > df["frequency"].quantile(0.8)) & (df["p_churn_30day"] > 0.6))
                with st.expander(f"🔥 Binge-and-Burnout Prevention ({len(burnout_list)})"):
                    st.caption("High-intensity users at risk of exhaustion. Offering high-retention series.")
                    st.dataframe(burnout_list, use_container_width=True, hide_index=True)
        else:
            st.info("No prediction data available to generate attention lists.")
            
        st.markdown("---")
            
        if "CLTV Predictions" in data:
            df_preds = data["CLTV Predictions"]
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_preds, x="clv_12mo", 
                                    nbins=30, title="Distribution of Predicted CLTV",
                                    color_discrete_sequence=['#667eea'])
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
        
            with col2:
                if "clv_segment" in df_preds.columns:
                    segment_counts = df_preds["clv_segment"].value_counts().reset_index()
                    fig_pie = px.pie(segment_counts, values="count", names="clv_segment", 
                                    title="Customer Segments Distribution",
                                    hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                    fig_pie.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

            # Bubble Plot in Full Width
            st.subheader("💡 Deep Dive: Frequency vs CLTV Bubble Analysis")
            fig_bubble = px.scatter(df_preds, x="frequency", y="clv_12mo",
                                color="clv_segment" if "clv_segment" in df_preds.columns else "expected_transactions_12mo", 
                                size="monetary_value",
                                hover_data=["customer_id"],
                                title="Customer Value Matrix (Bubble size = Monetary Value)",
                                template="plotly_white",
                                height=600,
                                opacity=0.7)
            fig_bubble.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_bubble, use_container_width=True)

    # --- TAB 2: CUSTOMER INSIGHTS ---
    with tabs[1]:
        st.header("🎯 CLTV & Churn Predictions Analysis")
        if "CLTV Predictions" in data:
            df_preds = data["CLTV Predictions"]
            
            # --- Detailed Customer Lookup ---
            customer_ids = sorted(df_preds["customer_id"].unique())
            selected_cust_id = st.selectbox("Select Customer ID", customer_ids, key="customer_select_box")
            cust_record = df_preds[df_preds["customer_id"] == selected_cust_id].iloc[0]

            # Row 1: Prediction Card & AI Analysis
            col_card, col_analysis = st.columns([1, 1])
            
            with col_card:
                # Show Visual Insight Card
                st.markdown("##### 🚀 Prediction Intelligence Card")
                
                # Color logic for churn risk
                p_alive = cust_record.get('p_alive', 0)
                status_color = "#28a745" if p_alive > 0.8 else "#ffc107" if p_alive > 0.5 else "#dc3545"
                status_label = "HEALTHY" if p_alive > 0.8 else "AT RISK" if p_alive > 0.5 else "CRITICAL CHURN"
                accuracy = cust_record.get('model_accuracy_metric', 0)
                
                html_card = f"""
                <div class='customer-card' style='background: white; padding: 25px; border-radius: 20px; border-left: 10px solid {status_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 0.8rem; color: #667eea; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>{cust_record.get('cohort_month', 'N/A')} Cohort</span>
                        <div style='display: flex; align-items: center; gap: 8px;'>
                            <span style='font-size: 0.7rem; font-weight: 800; color: {status_color}; letter-spacing: 0.5px;'>{status_label}</span>
                            <span style='background: {status_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;'>{cust_record.get('clv_segment', 'N/A')}</span>
                        </div>
                    </div>
                    <h3 style='margin: 15px 0 5px 0; color: #2c3e50; font-size: 1.4rem;'>Customer {selected_cust_id}</h3>
                    <p style='color: #7f8c8d; font-size: 0.85rem; margin-bottom: 20px;'>Prediction generated on: {cust_record.get('prediction_date', 'N/A')}</p>
                    
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;'>
                        <div style='background: #f8f9fa; padding: 12px; border-radius: 12px; border: 1px solid #eee; cursor: help;'>
                            <div title='{TOOLTIPS["churn_risk"]}' style='font-size: 0.7rem; color: #667eea; font-weight: 600; text-transform: uppercase;'>Churn Risk (30d) ⓘ</div>
                            <div style='font-size: 1.2rem; font-weight: 700; color: #dc3545;'>{cust_record.get('p_churn_30day', 0):.1%}</div>
                        </div>
                        <div style='background: #f8f9fa; padding: 12px; border-radius: 12px; border: 1px solid #eee; cursor: help;'>
                            <div title='{TOOLTIPS["p_alive"]}' style='font-size: 0.7rem; color: #667eea; font-weight: 600; text-transform: uppercase;'>P(Alive) ⓘ</div>
                            <div style='font-size: 1.2rem; font-weight: 700; color: #28a745;'>{p_alive:.1%}</div>
                        </div>
                    </div>

                    <div style='margin-bottom: 20px;'>
                        <div style='font-size: 0.75rem; color: #2c3e50; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;'>CLTV Projections</div>
                        <div style='display: flex; justify-content: space-between;'>
                            <div title='{TOOLTIPS["clv_12mo"]}' style='cursor: help;'>
                                <div style='font-size: 0.65rem; color: #7f8c8d;'>12 Months ⓘ</div>
                                <div style='font-size: 1rem; font-weight: 700;'>${cust_record.get('clv_12mo', 0):,.1f}</div>
                            </div>
                            <div title='{TOOLTIPS["clv_24mo"]}' style='cursor: help;'>
                                <div style='font-size: 0.65rem; color: #7f8c8d;'>24 Months ⓘ</div>
                                <div style='font-size: 1rem; font-weight: 700;'>${cust_record.get('clv_24mo', 0):,.1f}</div>
                            </div>
                            <div title='{TOOLTIPS["clv_36mo"]}' style='cursor: help;'>
                                <div style='font-size: 0.65rem; color: #7f8c8d;'>36 Months ⓘ</div>
                                <div style='font-size: 1rem; font-weight: 700;'>${cust_record.get('clv_36mo', 0):,.1f}</div>
                            </div>
                        </div>
                    </div>

                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;'>
                         <div title='{TOOLTIPS["expected_tx"]}' style='cursor: help;'>
                            <div style='font-size: 0.65rem; color: #7f8c8d; font-weight: 600;'>Expected Transactions (12m) ⓘ</div>
                            <div style='font-size: 0.9rem; font-weight: 700;'>{cust_record.get('expected_transactions_12mo', 0):.1f}</div>
                        </div>
                        <div title='{TOOLTIPS["avg_tx_val"]}' style='cursor: help;'>
                            <div style='font-size: 0.65rem; color: #7f8c8d; font-weight: 600;'>Avg. Transaction Value ⓘ</div>
                            <div style='font-size: 0.9rem; font-weight: 700;'>${cust_record.get('expected_value_per_transaction', 0):,.1f}</div>
                        </div>
                    </div>

                    <div title='{TOOLTIPS["ci"]}' style='background: rgba(102, 126, 234, 0.03); padding: 12px; border-radius: 12px; margin-bottom: 20px; cursor: help;'>
                        <div style='font-size: 0.65rem; color: #7f8c8d; font-weight: 600;'>95% Confidence Interval ⓘ</div>
                        <div style='font-size: 0.85rem; font-weight: 700; color: #667eea;'>
                            ${cust_record.get('clv_95_ci_lower', 0):,.0f} &mdash; ${cust_record.get('clv_95_ci_upper', 0):,.0f}
                        </div>
                    </div>
                    
                    <div title='{TOOLTIPS["model_conf"]}'>
                         <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                            <span style='font-size: 0.75rem; color: #7f8c8d; font-weight: 600;'>Model Confidence ⓘ</span>
                            <span style='font-size: 0.75rem; font-weight: 700; color: #667eea;'>{accuracy:.1%}</span>
                        </div>
                        <div style='background: #ecf0f1; height: 8px; border-radius: 4px; overflow: hidden;'>
                            <div style='background: #667eea; width: {accuracy*100}%; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                </div>
                """
                st.html(html_card)

            with col_analysis:
                # --- AI INTERPRETATION SECTION ---
                st.markdown("##### 🤖 AI Customer Health Analysis")
                interpretation = load_customer_interpretation(selected_cust_id)
                st.markdown(f"""
                <div class='ai-report-container'>
                    <div class='ai-report-header'>
                        <span class='ai-report-badge'>AI Diagnostics</span>
                        <span style='font-size: 0.8rem; color: #7f8c8d; font-weight: 600;'>System Analysis</span>
                    </div>
                    <div class='ai-report-content'>
                        {interpretation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Row 2: Customer Profile Details
            st.divider()
            st.markdown(f"### Customer Profile: {selected_cust_id}")
            
            # Top metrics ROW 1
            m1, m2, m3 = st.columns(3)
            m1.metric("CLV (12 Mo)", f"${cust_record.get('clv_12mo', 0):,.2f}", 
                        delta=f"Segment: {cust_record.get('clv_segment', 'N/A')}",
                        help=TOOLTIPS["clv_12mo"])
            m2.metric("P(Alive)", f"{cust_record.get('p_alive', 0):.1%}",
                        help=TOOLTIPS["p_alive"])
            m3.metric("Churn Risk (30d)", f"{cust_record.get('p_churn_30day', 0):.1%}", 
                        delta_color="inverse", delta="High Risk" if cust_record.get('p_churn_30day', 0) > 0.5 else "Low Risk",
                        help=TOOLTIPS["churn_risk"])
            
            st.divider()
            
            # Details ROW 2
            d1, d2, d3 = st.columns(3)
            d1.info(f"**Expected tx (12mo):** {cust_record.get('expected_transactions_12mo', 0):.2f}")
            d2.info(f"**Avg Value/Tx:** ${cust_record.get('expected_value_per_transaction', 0):.2f}")
            d3.info(f"**95% CI:** ${cust_record.get('clv_95_ci_lower',0):.0f} - ${cust_record.get('clv_95_ci_upper',0):.0f}")
            
        else:
            st.warning("No CLTV predictions found. Run the data science pipeline.")

    # --- TAB 3: MODEL EXPLORER ---
    with tabs[2]:
        st.header("🧬 Interactive Customer Simulator")
        st.markdown("Simulate predictions for a specific customer based on their history.")
        
        if "bg_nbd" in models and "gamma_gamma" in models:
            bgf = models["bg_nbd"]
            ggf = models["gamma_gamma"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Customer Characteristics")
                freq_input = st.number_input("Frequency (Repeat purchases)", 0, 1000, 5)
                rec_input = st.number_input("Recency (Time between first and last purchase)", 0, 1000, 30)
                t_input = st.number_input("T (Age of customer since first purchase)", 0, 1000, 100)
                mon_input = st.number_input("Monetary Value (Avg. revenue per purchase)", 0.0, 10000.0, 100.0)
                period = st.slider("Prediction Horizon (Months)", 1, 36, 12)
            
            with col2:
                st.subheader("Prediction Outcome")
                # BG/NBD prediction
                expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                    period, freq_input, rec_input, t_input
                )
                
                # Gamma-Gamma prediction (Conditional Expected Average Profit)
                if freq_input > 0:
                    expected_value = ggf.conditional_expected_average_profit(freq_input, mon_input)
                    # lifetimes expectation: customer_lifetime_value often expects Series to handle indices
                    cltv_result = ggf.customer_lifetime_value(
                        bgf, 
                        pd.Series([freq_input]), 
                        pd.Series([rec_input]), 
                        pd.Series([t_input]), 
                        pd.Series([mon_input]), 
                        time=period
                    )
                    total_cltv = cltv_result.iloc[0] if not isinstance(cltv_result, (int, float)) else cltv_result
                else:
                    expected_value = 0
                    total_cltv = 0

                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.6); padding: 30px; border-radius: 15px; border: 2px solid #667eea; box-shadow: 0 10px 25px rgba(0,0,0,0.05);'>
                    <h2 style='margin:0; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Results for {period} Months</h2>
                    <hr style='border-top: 1px solid rgba(0,0,0,0.1);'>
                    <p title='{TOOLTIPS["expected_tx"]}' style='font-size: 1.2rem; color: #2c3e50; cursor: help;'>Expected Purchases: <b>{expected_purchases:.2f} ⓘ</b></p>
                    <p title='{TOOLTIPS["avg_tx_val"]}' style='font-size: 1.2rem; color: #2c3e50; cursor: help;'>Cond. Expected Avg Profit: <b>${expected_value:.2f} ⓘ</b></p>
                    <h1 title='Projected customer value for this period' style='background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 20px; cursor: help;'>CLTV: ${total_cltv:.2f} ⓘ</h1>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Models not loaded. Please ensure models are trained and saved in data/06_models/")

    # --- TAB 4: MODEL INSIGHTS ---
    with tabs[3]:
        st.header("📈 Model Diagnostics & Parameters")
        
        # AI Interpretation for Model Insights
        if "cltv_overview" in interpretations:
            with st.expander("🤖 AI Strategic Interpretation", expanded=False):
                st.markdown(interpretations["cltv_overview"])

        # SLM Interpretation for Model Parameters
        if "model_parameters" in interpretations:
            with st.expander("🧠 AI Parameter Explanation (SLM)", expanded=False):
                st.markdown(interpretations["model_parameters"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BG/NBD Model Parameters")
            if "bg_nbd" in models:
                params = models["bg_nbd"].params_
                st.table(pd.DataFrame(params, columns=["Value"]))
                st.info("The BG/NBD model captures the purchase frequency and churn probability.")
            else:
                st.write("BG/NBD model not loaded.")
                
        with col2:
            st.subheader("Gamma-Gamma Model Parameters")
            if "gamma_gamma" in models:
                params = models["gamma_gamma"].params_
                st.table(pd.DataFrame(params, columns=["Value"]))
                st.info("The Gamma-Gamma model predicts the monetary value of future transactions.")
            else:
                st.write("Gamma-Gamma model not loaded.")

        st.markdown("---")
        st.markdown("""
        ### Understanding the Models
        - **BG/NBD Model**: Predicts the 'if' and 'when' of future transactions. It models the purchase rate (lambda) and the dropout rate (p).
        - **Gamma-Gamma Model**: Predicts the 'how much' of future transactions. It relies on the assumption that monetary value and purchase frequency are independent.
        - **CLTV**: The product of the expected number of transactions (from BG/NBD) and the expected average transaction value (from Gamma-Gamma).
        """)

    # --- TAB 5: STRATEGIC KPIS ---
    with tabs[4]:
        st.header("🚀 Strategic Application & Business Impact")
        
        # AI Interpretation for Strategic KPIs
        if "churn_risk_analysis" in interpretations:
            with st.expander("🤖 AI Churn & Retention Analysis", expanded=False):
                st.markdown(interpretations["churn_risk_analysis"])
        
        # Paths to generated plots
        PLOT_DIR = "data/08_reporting/plots"
        img_revenue = os.path.join(PLOT_DIR, "kpi_revenue_per_customer.png")
        img_acquisition = os.path.join(PLOT_DIR, "kpi_acquisition_efficiency.png")
        img_segments = os.path.join(PLOT_DIR, "kpi_revenue_by_segment.png")
        img_churn = os.path.join(PLOT_DIR, "kpi_churn_risk_dist.png")
        
        # Row 1: High Level Strategy
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue per Customer Impact")
            if os.path.exists(img_revenue):
                st.image(img_revenue, use_container_width=True)
                st.caption("Comparison of Average CLTV vs. High-Value Segment. Targeting high-value users can significantly boost revenue per capita.")
            else:
                st.info("Revenue per Customer plot not found. Run visualization pipeline.")

        with col2:
            st.subheader("Acquisition Efficiency (LTV:CAC)")
            if os.path.exists(img_acquisition):
                st.image(img_acquisition, use_container_width=True)
                st.caption("Ratio of Customer Lifetime Value to Customer Acquisition Cost. A ratio > 3:1 is typically healthy for SaaS/Subscription models.")
            else:
                st.info("Acquisition Efficiency plot not found. Run visualization pipeline.")

        st.divider()

        # Row 2: Deep Dive
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Revenue Concentration by Segment")
            if os.path.exists(img_segments):
                st.image(img_segments, use_container_width=True)
                st.caption("Total revenue potential distributed by CLTV Segment quintiles.")
            else:
                st.info("Segment plot not available.")

        with col4:
            st.subheader("Churn Risk Profile (Next 30 Days)")
            if os.path.exists(img_churn):
                st.image(img_churn, use_container_width=True)
                st.caption("Distribution of customers by probability of churn/inactivity in the next 30 days.")
            else:
                st.info("Churn plot not available.")


        st.markdown("---")
        st.subheader("Business Scenarios Simulation")
        
        col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
        
        # Calculate dynamic metrics if predictions are available
        if "CLTV Predictions" in data:
            df = data["CLTV Predictions"]
            clv_col = "clv_12mo" if "clv_12mo" in df.columns else "predicted_cltv"
            if clv_col in df.columns: 
                avg_cltv = df[clv_col].mean()
                total_customers = len(df)
                
                # Scenario: Churn Reduction Impact
                # If we save 1% of churn, how much value is that?
                churn_save_rate = 0.01
                value_saved = df[clv_col].sum() * churn_save_rate
                
                with col_metrics_1:
                    st.metric("Avg. CLTV", f"${avg_cltv:.2f}", help=TOOLTIPS["clv_12mo"])
                
                with col_metrics_2:
                    st.metric("Value of 1% Churn Reduction", f"${value_saved:,.0f}", delta="Saved Revenue", help=TOOLTIPS["churn_impact"])
                    st.caption(f"Impact of retaining {int(total_customers * 0.01)} customers")

                with col_metrics_3:
                    # Scenario: High Value Segment
                    if "clv_segment" in df.columns:
                        count_top = len(df[df["clv_segment"] == "Q5_Elite"])
                        st.metric("Elite Customers (Q5)", f"{count_top:,}", help=TOOLTIPS["elite_count"])
                    else:
                        top_20_percent_val = df[clv_col].quantile(0.8)
                        count_top = len(df[df[clv_col] >= top_20_percent_val])
                        st.metric("High Value Customers (Top 20%)", f"{count_top:,}", help=TOOLTIPS["elite_count"])
                    st.caption("Most valuable customer segment count")

if __name__ == "__main__":
    main()