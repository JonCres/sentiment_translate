import streamlit as st
import pandas as pd
from datetime import datetime

# Import custom components and modules
# Assumes app/ is the root in sys.path (standard Streamlit behavior)
from styles import apply_custom_styles
from utils import load_real_data, load_params
from components.sidebar import render_sidebar
from components.executive_summary import render_executive_summary
from components.customer_sentiment import render_customer_sentiment
from components.key_insights import render_key_insights
from components.strategic_intelligence import render_strategic_intelligence
from components.account_intelligence import render_account_intelligence
from components.voice_of_client import render_voice_of_client
from components.action_center import render_action_center
from components.temporal_analysis import render_temporal_analysis

# --- Page Configuration ---
st.set_page_config(
    page_title="Voice of the Client Analytics | Marketing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Apply Styling ---
apply_custom_styles()

# --- Data Loading ---
real_df = load_real_data()

# --- Sidebar & Navigation ---
selected_page = render_sidebar(real_df)

# --- Main Content Routing ---
if selected_page == "🏠 Executive Summary":
    render_executive_summary(real_df)

elif selected_page == "📈 Customer Sentiment":
    render_customer_sentiment(real_df)

elif selected_page == "💡 Key Insights & Themes":
    render_key_insights(real_df)

elif selected_page == "🧠 Strategic Intelligence":
    render_strategic_intelligence(real_df)

elif selected_page == "🎯 Account Intelligence":
    render_account_intelligence(real_df)

elif selected_page == "🗣️ Voice of the Client":
    render_voice_of_client(real_df)

elif selected_page == "⚡ Action Center":
    render_action_center(real_df)

elif selected_page == "📅 Temporal Intelligence":
    render_temporal_analysis(real_df)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: #5a6c7d;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            💡 Powered by AI-driven customer intelligence • Last updated: {date}
        </p>
        <p style='margin: 8px 0 0 0; font-size: 0.85rem;'>
            Questions? Contact your Customer Success team
        </p>
    </div>
    """.format(date=datetime.now().strftime("%B %d, %Y")),
    unsafe_allow_html=True
)