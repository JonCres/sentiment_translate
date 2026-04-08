import streamlit as st
from logic import get_portfolio_data, get_sentiment_insights


def navigate_to_topic(topic_name):
    st.session_state.sidebar_menu = "🗣️ Voice of the Client"
    st.session_state.topic_filter = topic_name
    # Reset other filters to default
    st.session_state.voc_account_filter = ["All"]
    st.session_state.voc_emotion_filter = ["All"]
    st.session_state.voc_sentiment_filter = "All"
    st.session_state.voc_search_text = ""
    st.rerun()


def navigate_to_page(page_name):
    st.session_state.sidebar_menu = page_name
    st.rerun()


def render_sidebar(real_df):
    if "sidebar_menu" not in st.session_state:
        st.session_state.sidebar_menu = "🏠 Executive Summary"

    if "topic_filter" not in st.session_state:
        st.session_state.topic_filter = None

    with st.sidebar:
        # ... existing image and title ...
        st.markdown(
            """
            <div style='text-align: center; padding: 20px 0;'>
                <h2 style='color: #00a8cc; margin: 0;'>📊 Voice of the Client</h2>
                <p style='color: #5a6c7d; margin-top: 8px; font-size: 0.9rem;'>Marketing Intelligence Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        menu_options = [
            "🏠 Executive Summary",
            "📈 Customer Sentiment",
            "💡 Key Insights & Themes",
            "🧠 Strategic Intelligence",
            "🎯 Account Intelligence",
            "🗣️ Voice of the Client",
            "⚡ Action Center",
            "📅 Temporal Intelligence",
        ]

        # Sync widget state with session state
        if st.session_state.sidebar_menu not in menu_options:
            st.session_state.sidebar_menu = menu_options[0]

        menu = st.radio(
            "Navigate",
            menu_options,
            index=menu_options.index(st.session_state.sidebar_menu),
            label_visibility="collapsed",
            key="sidebar_nav_radio",  # Changed key to avoid conflict if any, though session_state handling is manual
            on_change=lambda: st.session_state.update(
                {"sidebar_menu": st.session_state.sidebar_nav_radio}
            ),
        )

        st.markdown("---")

        # Quick Stats in Sidebar
        portfolio = get_portfolio_data(real_df)
        sentiment_data = get_sentiment_insights(real_df)

        st.markdown("### 📊 Quick Stats")

        if not portfolio.empty:
            critical_count = len(portfolio[portfolio["Status"] == "At Risk"])
            st.metric(
                "Accounts at Risk",
                critical_count,
                delta=f"-{critical_count} to address",
                delta_color="inverse",
            )

        if sentiment_data:
            st.metric("NPS", f"{sentiment_data['nps']}")

        st.markdown("---")
        st.markdown(
            """
            <div style='padding: 16px; background: #e3f2fd; border-radius: 8px; border: 1px solid #90caf9;'>
                <p style='margin: 0; font-size: 0.85rem; color: #1565c0;'>
                    💡 <strong>Tip:</strong> Use filters to drill down into specific accounts, time periods, or sentiment categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.sidebar_menu
