import streamlit as st
import pandas as pd
from datetime import datetime
from utils import get_ai_interpretation
from logic import get_portfolio_data
from components.sidebar import navigate_to_page


def render_action_center(real_df):
    st.title("⚡ Action Center")
    st.markdown("*Prioritized recommendations for customer success*")
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
                key="action_date_filter",
            )

        if selected_date != "Overall":
            real_df = real_df[dates_series == selected_date]
    # -------------------

    st.markdown("### 🤖 AI Insight Interpretation")
    st.info(get_ai_interpretation("analysis_drivers", "nps_norm", type="groq"))
    st.markdown("---")

    portfolio = get_portfolio_data(real_df)

    # High-priority accounts
    st.subheader("🚨 Immediate Attention Required")

    if not portfolio.empty:
        critical_accounts = portfolio[portfolio["Status"] == "At Risk"].sort_values(
            "Health Score"
        )

        if not critical_accounts.empty:
            for _, acc in critical_accounts.iterrows():
                with st.expander(
                    f"🔴 {acc['Account Name']} - Health Score: {acc['Health Score']}/100",
                    expanded=False,
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 📊 Account Status")
                        st.markdown(f"**Tier:** {acc['Tier']}")
                        st.markdown(f"**Average NPS:** {acc['Avg NPS']}")
                        st.markdown(f"**Trend:** {acc['Trend']}")
                        st.markdown(f"**Feedback Count:** {acc['Feedback Count']}")

                        st.markdown("---")
                        st.markdown("### 🎯 Recommended Actions")

                        actions = [
                            ("Schedule 1:1 executive review call", "High"),
                            ("Analyze recent support tickets", "High"),
                            ("Send personalized re-engagement campaign", "Medium"),
                            ("Review product usage patterns", "Medium"),
                            ("Check contract renewal timeline", "High"),
                        ]

                        for action, priority in actions:
                            priority_color = (
                                "#FF5252" if priority == "High" else "#FFA726"
                            )
                            completed = st.checkbox(
                                f"{action}",
                                key=f"{acc['Account Name']}_{action}",
                                help=f"Priority: {priority}",
                            )

                    with col2:
                        st.markdown("### 📅 Quick Actions")

                        if st.button(
                            "📞 Log Call",
                            key=f"call_{acc['Account Name']}",
                            use_container_width=True,
                        ):
                            st.success("Call logged successfully!")

                        if st.button(
                            "📧 Send Email",
                            key=f"email_{acc['Account Name']}",
                            use_container_width=True,
                        ):
                            st.success("Email template opened!")

                        if st.button(
                            "📝 Add Note",
                            key=f"note_{acc['Account Name']}",
                            use_container_width=True,
                        ):
                            st.text_area(
                                "Note",
                                key=f"note_text_{acc['Account Name']}",
                                height=100,
                            )

                        if st.button(
                            "✅ Mark Resolved",
                            key=f"resolve_{acc['Account Name']}",
                            use_container_width=True,
                        ):
                            st.success("Status updated!")
        else:
            st.success("🎉 No critical accounts! All customers are in good health.")
    else:
        st.info("No data available to determine critical accounts.")

    st.markdown("---")

    # Accounts needing attention
    st.subheader("⚠️ Proactive Engagement Opportunities")

    if not portfolio.empty:
        attention_accounts = portfolio[
            portfolio["Status"] == "Needs Attention"
        ].sort_values("Health Score")

        if not attention_accounts.empty:
            for _, acc in attention_accounts.head(5).iterrows():
                with st.expander(
                    f"🟠 {acc['Account Name']} - Health Score: {acc['Health Score']}/100"
                ):
                    st.markdown(
                        f"**Suggested Action:** Proactive check-in to maintain positive momentum"
                    )
                    st.markdown(
                        f"**Current NPS:** {acc['Avg NPS']} | **Trend:** {acc['Trend']}"
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.button("Schedule QBR", key=f"qbr_{acc['Account Name']}")
                    col2.button(
                        "Share Success Story", key=f"success_{acc['Account Name']}"
                    )
                    col3.button(
                        "Product Training", key=f"training_{acc['Account Name']}"
                    )
