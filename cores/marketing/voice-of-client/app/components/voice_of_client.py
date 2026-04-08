import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_ai_interpretation, load_params


def render_voice_of_client(real_df):
    st.title("🗣️ Voice of the Client")
    st.markdown("*Direct customer feedback and verbatims*")
    st.markdown("---")

    if real_df is not None and not real_df.empty:
        # Advanced Filters
        st.subheader("🔍 Filter & Search")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            account_filter = st.multiselect(
                "Accounts",
                options=["All"]
                + sorted(real_df["Account_Name"].dropna().unique().tolist()),
                default=st.session_state.get("voc_account_filter", ["All"]),
                key="voc_account_filter",
            )

        with col2:
            if "dominant_emotion" in real_df.columns:
                emotion_filter = st.multiselect(
                    "Emotions",
                    options=["All"]
                    + sorted(real_df["dominant_emotion"].dropna().unique().tolist()),
                    default=st.session_state.get("voc_emotion_filter", ["All"]),
                    key="voc_emotion_filter",
                )
            else:
                emotion_filter = ["All"]

        with col3:
            sentiment_index = 0
            if "voc_sentiment_filter" in st.session_state:
                options = [
                    "All",
                    "Promoters (9-10)",
                    "Passives (7-8)",
                    "Detractors (0-6)",
                ]
                if st.session_state.voc_sentiment_filter in options:
                    sentiment_index = options.index(
                        st.session_state.voc_sentiment_filter
                    )

            sentiment_filter = st.selectbox(
                "Sentiment",
                ["All", "Promoters (9-10)", "Passives (7-8)", "Detractors (0-6)"],
                index=sentiment_index,
                key="voc_sentiment_filter",
            )

        with col4:
            search_text = st.text_input(
                "🔎 Search keywords",
                st.session_state.get("voc_search_text", ""),
                key="voc_search_text",
            )

        with col5:
            # New: Date Dropdown Filter
            selected_date = "Overall"
            if "Timestamp" in real_df.columns:
                # Robustly convert to datetime and drop invalid values to get unique dates
                ts_series = pd.to_datetime(
                    real_df["Timestamp"], errors="coerce"
                ).dropna()
                if not ts_series.empty:
                    unique_dates = sorted(
                        ts_series.dt.date.astype(str).unique().tolist(), reverse=True
                    )

                    selected_date = st.selectbox(
                        "📅 Select Date",
                        options=["Overall"] + unique_dates,
                        index=0,
                        key="voc_date_filter",
                    )

            row_count = st.selectbox(
                "Show Rows",
                options=[10, 20, 50, 100, "All"],
                index=1,
                key="voc_row_count",
            )

        # Initialize session state for pagination
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        # Apply filters
        filtered_df = real_df.copy()

        # New: Topic filtering from session state (programmatic navigation)
        if st.session_state.get("topic_filter"):
            st.info(f"📍 Filtering by Topic: **{st.session_state.topic_filter}**")
            if st.button("❌ Clear Topic Filter"):
                st.session_state.topic_filter = None
                st.rerun()
            filtered_df = filtered_df[
                filtered_df["Topic_Name"] == st.session_state.topic_filter
            ]

        # Apply Date Dropdown Filter
        if selected_date != "Overall":
            # Ensure Timestamp is datetime and filter
            filtered_df["_temp_date_str"] = pd.to_datetime(
                filtered_df["Timestamp"], errors="coerce"
            ).dt.date.astype(str)
            filtered_df = filtered_df[filtered_df["_temp_date_str"] == selected_date]
            filtered_df = filtered_df.drop(columns=["_temp_date_str"])

        if "All" not in account_filter:
            filtered_df = filtered_df[filtered_df["Account_Name"].isin(account_filter)]

        if "All" not in emotion_filter and "dominant_emotion" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["dominant_emotion"].isin(emotion_filter)
            ]

        if sentiment_filter != "All":
            if sentiment_filter == "Promoters (9-10)":
                filtered_df = filtered_df[filtered_df["NPS_Score"] >= 9]
            elif sentiment_filter == "Passives (7-8)":
                filtered_df = filtered_df[
                    (filtered_df["NPS_Score"] >= 7) & (filtered_df["NPS_Score"] < 9)
                ]
            else:
                filtered_df = filtered_df[filtered_df["NPS_Score"] < 7]

        if search_text:
            mask = filtered_df["Interaction_Payload"].str.contains(
                search_text, case=False, na=False
            ) | filtered_df["Improvement_Comments"].str.contains(
                search_text, case=False, na=False
            )
            filtered_df = filtered_df[mask]

        # Determine feedbacks per page
        feedbacks_per_page = int(row_count) if row_count != "All" else len(filtered_df)
        if feedbacks_per_page == 0:
            feedbacks_per_page = 1  # Avoid division by zero if filtered_df is empty

        # Recalculate total pages after all filters are applied
        total_feedbacks = len(filtered_df)
        total_pages = (total_feedbacks + feedbacks_per_page - 1) // feedbacks_per_page
        if total_pages == 0:
            total_pages = 1  # Ensure at least one page if no data

        # Adjust current page if it's out of bounds
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = total_pages
        if st.session_state.current_page < 1:
            st.session_state.current_page = 1

        # Pagination controls - TOP
        col_prev_top, col_page_info_top, col_next_top = st.columns([1, 2, 1])
        with col_prev_top:
            if st.button(
                "⬅️ Previous",
                key="prev_page_top",
                disabled=st.session_state.current_page == 1,
            ):
                st.session_state.current_page -= 1
                st.rerun()
        with col_page_info_top:
            st.markdown(
                f"<p style='text-align: center; font-size: 1.1rem; font-weight: 500;'>Page {st.session_state.current_page} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with col_next_top:
            if st.button(
                "Next ➡️",
                key="next_page_top",
                disabled=st.session_state.current_page == total_pages,
            ):
                st.session_state.current_page += 1
                st.rerun()

        st.markdown("---")

        st.markdown("### 🤖 AI Insight Interpretation")
        st.info(
            get_ai_interpretation("feature_distributions", "NPS_Score", type="groq")
        )

        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"📋 {len(filtered_df)} Customer Responses")

        with col2:
            export_btn = st.button("📥 Export to CSV", use_container_width=True)
            if export_btn and not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "customer_feedback.csv",
                    "text/csv",
                    use_container_width=True,
                )

        # Calculate start and end indices for the current page
        start_idx = (st.session_state.current_page - 1) * feedbacks_per_page
        end_idx = start_idx + feedbacks_per_page

        display_df = filtered_df.iloc[start_idx:end_idx]

        # Display feedback cards
        st.markdown("---")

        for idx, row in display_df.iterrows():
            # Determine sentiment badge
            nps = row.get("NPS_Score", 0)
            if nps >= 9:
                sentiment_badge = '<span style="background: #66BB6A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">😊 Promoter</span>'
            elif nps >= 7:
                sentiment_badge = '<span style="background: #FFA726; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">😐 Passive</span>'
            else:
                sentiment_badge = '<span style="background: #FF5252; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">😞 Detractor</span>'

            # Emotion badge
            emotion = row.get("dominant_emotion", "N/A")
            emotion_emoji = {
                "joy": "😊",
                "sadness": "😢",
                "anger": "😠",
                "fear": "😨",
                "surprise": "😲",
                "neutral": "😐",
            }.get(emotion.lower(), "💭")

            # Load formatting from parameters
            params = load_params()
            ts_format = (
                params.get("data_processing", {})
                .get("skeleton_mapping", {})
                .get("defaults", {})
                .get("Timestamp_Format", "%Y-%m-%d")
            )

            ts_val = row.get("Timestamp")
            if pd.notna(ts_val):
                try:
                    date_str = pd.to_datetime(ts_val).strftime(ts_format)
                except Exception:
                    date_str = str(ts_val)[:10]
            else:
                date_str = "Unknown"
            interaction_id = row.get("Interaction_ID", "N/A")
            csat_val = row.get("CSAT_Score")
            csat_str = f" • CSAT: {int(csat_val)}" if pd.notna(csat_val) else ""

            with st.expander(
                f"[{interaction_id}] {row.get('Account_Name', 'Unknown')} • Date: {date_str} • NPS: {nps}{csat_str}",
                expanded=False,
            ):
                # Header with badges
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                        {sentiment_badge}
                        <span style="background: #e3f2fd; color: #0077aa; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">{emotion_emoji} {emotion.title()}</span>
                        <span style="background: #f5f5f5; color: #5a6c7d; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; border: 1px solid #d1d9e0;">{row.get("Language", "EN")}</span>
                        <span style="background: #fff3e0; color: #e65100; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">🏷️ {row.get("Topic_Name", "Uncategorized")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # feedback content and AI insight
                col_text, col_insight = st.columns([3, 2])

                with col_text:
                    st.markdown("**🇬🇧 English Feedback**")
                    english_feedback = row.get(
                        "Interaction_Payload", "No English feedback provided"
                    )
                    st.markdown(
                        f'<div class="customer-quote">{english_feedback}</div>',
                        unsafe_allow_html=True,
                    )

                    # Display original language feedback if it exists and is not null/empty
                    original_lang = row.get("detected_language", "en")
                    original_payload = row.get("Original_Interaction_Payload")

                    # Show original if it exists and is not empty
                    if (
                        pd.notna(original_payload)
                        and str(original_payload).strip() != ""
                    ):
                        display_lang = (
                            original_lang.upper() if original_lang else "ORIGINAL"
                        )
                        st.markdown(f"**🌐 Original Language ({display_lang})**")
                        st.markdown(
                            f'<div class="customer-quote">{original_payload}</div>',
                            unsafe_allow_html=True,
                        )

                with col_insight:
                    st.markdown("**🤖 AI Strategic Insight**")
                    # Use interaction_id as key for row-level Groq insight
                    insight = get_ai_interpretation(
                        "feedback_insights", interaction_id, type="groq"
                    )
                    st.info(insight)

                st.markdown("---")
                st.markdown("**📊 Model Outcomes**")

                # Check for predictive data presence
                has_predictions = any(
                    col in row.index and pd.notna(row[col])
                    for col in [
                        "Predicted_NPS_Score",
                        "Predicted_CSAT_Score",
                        "Predicted_Churn_Prob",
                    ]
                )

                # Detailed Outcomes in tabs
                tab_labels = ["🎯 Sentiment & Topics", "🎭 Emotions"]
                if has_predictions:
                    tab_labels.append("📈 Predictions")

                m_tabs = st.tabs(tab_labels)

                with m_tabs[0]:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Aspect-Based Sentiment**")
                        aspect_cols = [c for c in row.index if c.startswith("sent_")]
                        if aspect_cols:
                            for ac in aspect_cols:
                                a_name = ac.replace("sent_", "").title()
                                a_val = row[ac]
                                a_color = (
                                    "green"
                                    if a_val > 3.5
                                    else "orange"
                                    if a_val > 2.5
                                    else "red"
                                )
                                st.markdown(
                                    f"- {a_name}: <span style='color:{a_color}'>{a_val:.2f}/5</span>",
                                    unsafe_allow_html=True,
                                )
                    with c2:
                        st.markdown("**Context-Aware Sentiment**")
                        st.markdown(
                            f"- **Polarity:** {row.get('sentiment_polarity', 'N/A')}"
                        )
                        st.markdown(
                            f"- **Intensity:** {row.get('sentiment_intensity', 'N/A')}"
                        )
                        st.markdown(
                            f"- **Context Score:** {row.get('context_score', 'N/A')}"
                        )
                        st.markdown(f"**Topic ID:** {row.get('Topic_ID', 'N/A')}")

                with m_tabs[1]:
                    emo_cols = [
                        c
                        for c in row.index
                        if c.startswith("emo_") and not c.endswith("_raw")
                    ]
                    if emo_cols:
                        # Show top 3 emotions
                        emo_series = row[emo_cols].sort_values(ascending=False).head(3)
                        fig_row_emo = px.bar(
                            x=emo_series.values,
                            y=[e.replace("emo_", "").title() for e in emo_series.index],
                            orientation="h",
                            labels={"x": "Confidence", "y": ""},
                            height=200,
                            template="plotly_white",
                        )
                        fig_row_emo.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(
                            fig_row_emo,
                            use_container_width=True,
                            key=f"emotion_chart_{interaction_id}",
                        )
                    else:
                        st.info("No emotion data available.")

                if has_predictions:
                    with m_tabs[2]:
                        sc1, sc2, sc3 = st.columns(3)
                        if "Predicted_NPS_Score" in row.index:
                            sc1.metric(
                                "Predicted NPS",
                                f"{row.get('Predicted_NPS_Score', 0):.1f}",
                            )
                        if "Predicted_CSAT_Score" in row.index:
                            sc2.metric(
                                "Predicted CSAT",
                                f"{row.get('Predicted_CSAT_Score', 0):.1f}",
                            )

                        if "Predicted_Churn_Prob" in row.index:
                            churn_prob = row.get("Predicted_Churn_Prob", 0)
                            risk_label = row.get("Churn_Risk_Level", "Low")
                            risk_color = (
                                "red"
                                if risk_label == "High"
                                else "orange"
                                if risk_label == "Medium"
                                else "green"
                            )
                            st.markdown(
                                f"**Churn Risk:** <span style='color:{risk_color}; font-weight:bold;'>{risk_label}</span> ({churn_prob:.2%})",
                                unsafe_allow_html=True,
                            )

        # Pagination controls - BOTTOM
        st.markdown("---")
        col_prev_bottom, col_page_info_bottom, col_next_bottom = st.columns([1, 2, 1])
        with col_prev_bottom:
            if st.button(
                "⬅️ Previous",
                key="prev_page_bottom",
                disabled=st.session_state.current_page == 1,
            ):
                st.session_state.current_page -= 1
                st.rerun()
        with col_page_info_bottom:
            st.markdown(
                f"<p style='text-align: center; font-size: 1.1rem; font-weight: 500;'>Page {st.session_state.current_page} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with col_next_bottom:
            if st.button(
                "Next ➡️",
                key="next_page_bottom",
                disabled=st.session_state.current_page == total_pages,
            ):
                st.session_state.current_page += 1
                st.rerun()
