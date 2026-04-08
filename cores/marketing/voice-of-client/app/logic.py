import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import re
from typing import Dict, Any


def get_portfolio_data(real_df: pd.DataFrame):
    """Generate portfolio overview with account health metrics."""
    if real_df is None or real_df.empty or "Account_Name" not in real_df.columns:
        return pd.DataFrame()

    accounts = sorted(real_df["Account_Name"].dropna().unique().tolist())
    data = []

    for acc in accounts:
        # Explicitly create a copy to avoid SettingWithCopyWarning later
        acc_real = real_df[real_df["Account_Name"] == acc].copy()

        if not acc_real.empty:
            avg_nps = (
                acc_real["NPS_Score"].mean() if "NPS_Score" in acc_real.columns else 0
            )
            avg_csat = (
                acc_real["CSAT_Score"].mean() if "CSAT_Score" in acc_real.columns else 0
            )

            # Normalize components to 0-100 scale
            nps_norm_score = avg_nps * 10
            csat_norm_score = (avg_csat - 1) / 4.0 * 100 if avg_csat > 0 else 0

            # Compute composite Health Score (Average of available metrics)
            if avg_nps > 0 and avg_csat > 0:
                health_score = int((nps_norm_score + csat_norm_score) / 2)
            elif avg_nps > 0:
                health_score = int(nps_norm_score)
            elif avg_csat > 0:
                health_score = int(csat_norm_score)
            else:
                health_score = 0

            feedback_count = len(acc_real)

            # Calculate trend (last 30 days vs previous 30 days)
            if (
                "Timestamp" in acc_real.columns
                and pd.notna(acc_real["Timestamp"]).any()
            ):
                acc_real["Timestamp"] = pd.to_datetime(acc_real["Timestamp"])
                recent = acc_real[
                    acc_real["Timestamp"]
                    >= (acc_real["Timestamp"].max() - pd.Timedelta(days=30))
                ]
                trend = (
                    "↗️"
                    if len(recent) > 0 and recent["NPS_Score"].mean() > avg_nps
                    else "↘️"
                )
            else:
                trend = "→"
        else:
            avg_nps = 0.0
            avg_csat = 0.0
            health_score = 0
            feedback_count = len(acc_real)
            trend = "→"

        # Determine status
        if health_score < 60:
            status = "At Risk"
            color = "#FF5252"
            priority = "High"
        elif health_score < 75:
            status = "Needs Attention"
            color = "#FFA726"
            priority = "Medium"
        else:
            status = "Healthy"
            color = "#66BB6A"
            priority = "Low"

        data.append(
            {
                "Account Name": acc,
                "Avg NPS": round(avg_nps, 1),
                "Avg CSAT": round(avg_csat, 1),
                "Status": status,
                "Color": color,
                "Priority": priority,
                "Tier": acc_real["Tier"].iloc[0]
                if "Tier" in acc_real.columns
                else "Standard",
                "Health Score": health_score,
                "Feedback Count": feedback_count,
                "Trend": trend,
            }
        )

    return pd.DataFrame(data)


def get_sentiment_insights(real_df: pd.DataFrame):
    """Generate marketing-focused sentiment insights."""
    if real_df is None or real_df.empty:
        return {}

    def get_sentiment(score):
        if score >= 9:
            return "Promoter"
        if score >= 7:
            return "Passive"
        return "Detractor"

    df = real_df.copy()
    if "NPS_Score" in df.columns:
        df["sentiment"] = df["NPS_Score"].apply(get_sentiment)
        promoters = len(df[df["sentiment"] == "Promoter"])
        passives = len(df[df["sentiment"] == "Passive"])
        detractors = len(df[df["sentiment"] == "Detractor"])
    else:
        promoters = 0
        passives = 0
        detractors = 0

    total = len(df)

    nps = ((promoters - detractors) / total * 100) if total > 0 else 0

    avg_polarity = (
        df["sentiment_polarity"].mean() if "sentiment_polarity" in df.columns else 0
    )
    avg_intensity = (
        df["sentiment_intensity"].mean() if "sentiment_intensity" in df.columns else 0
    )

    return {
        "nps": round(nps, 1),
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "total": total,
        "promoter_pct": round(promoters / total * 100, 1) if total > 0 else 0,
        "detractor_pct": round(detractors / total * 100, 1) if total > 0 else 0,
        "avg_polarity": round(avg_polarity, 2),
        "avg_intensity": round(avg_intensity, 2),
    }


def calculate_overall_health_score(overall_nps: float, overall_csat: float) -> int:
    """Calculates the overall health score based on aggregated NPS and CSAT scores."""
    # NPS Normalized: scales 0-10 to 0-100
    # The get_sentiment_insights already returns NPS in the -100 to 100 range, so no need to multiply by 10.
    # Instead, we need to map -100 to 100 to 0-100 scale.
    nps_normalized = (overall_nps + 100) / 2

    # CSAT Normalized: scales 1-5 to 0-100
    # CSAT is often averaged on a 1-5 scale, so (CSAT - 1) / 4 * 100
    csat_normalized = ((overall_csat - 1) / 4) * 100 if overall_csat > 0 else 0

    # Compute composite Health Score (Average of available metrics)
    if overall_nps is not None and overall_csat is not None and overall_csat > 0:
        health_score = int((nps_normalized + csat_normalized) / 2)
    elif overall_nps is not None:
        health_score = int(nps_normalized)
    elif overall_csat is not None and overall_csat > 0:
        health_score = int(csat_normalized)
    else:
        health_score = 0

    # Ensure score is within 0-100 range
    health_score = max(0, min(100, health_score))

    return health_score


def get_top_themes(real_df: pd.DataFrame):
    """Extract top high-sentiment and low-sentiment topics for marketing messaging.

    High-Sentiment Topics: Topics with the highest average NPS scores.
    Low-Sentiment Topics: Topics with the lowest average NPS scores.

    Returns:
        Dict with 'positive' (highest NPS) and 'negative' (lowest NPS) theme lists.
    """
    if real_df is None or real_df.empty or "Topic_Name" not in real_df.columns:
        return {"positive": [], "negative": []}

    # Filter out Uncategorized/Outliers for clearer messaging
    df_filtered = real_df[~real_df["Topic_Name"].isin(["Uncategorized", "Outlier"])]

    if df_filtered.empty:
        return {"positive": [], "negative": []}

    # Group by Topic and calculate mean NPS
    topic_sentiment = df_filtered.groupby("Topic_Name")["NPS_Score"].mean()

    # Ensure we have enough unique topics for proper differentiation
    n_topics = len(topic_sentiment)
    n_select = min(3, n_topics // 2)  # At most 3, but ensure no overlap

    if n_topics < 2:
        # Not enough topics to differentiate
        return {"positive": [], "negative": []}

    # Sort and select from opposite ends of the spectrum
    sorted_topics = topic_sentiment.sort_values()

    # Get LOW sentiment topics (smallest NPS scores) - sorted ascending
    low_sentiment_topics = sorted_topics.head(n_select)

    # Get HIGH sentiment topics (largest NPS scores) - sorted descending
    high_sentiment_topics = sorted_topics.tail(n_select).sort_values(ascending=False)

    # Ensure no overlap: remove any topics that appear in both lists
    low_set = set(low_sentiment_topics.index)
    high_set = set(high_sentiment_topics.index)
    overlap = low_set.intersection(high_set)

    if overlap:
        # Remove overlapping topics from the low sentiment list
        low_sentiment_topics = low_sentiment_topics.drop(list(overlap))

    # Build positive themes (HIGH sentiment)
    positive_themes = [
        {
            "theme": topic,
            "id": int(
                df_filtered[df_filtered["Topic_Name"] == topic]["Topic_ID"].iloc[0]
            ),
            "score": round(val, 1),
            "count": len(df_filtered[df_filtered["Topic_Name"] == topic]),
        }
        for topic, val in high_sentiment_topics.items()
    ]

    # Build negative themes (LOW sentiment)
    negative_themes = [
        {
            "theme": topic,
            "id": int(
                df_filtered[df_filtered["Topic_Name"] == topic]["Topic_ID"].iloc[0]
            ),
            "score": round(val, 1),
            "count": len(df_filtered[df_filtered["Topic_Name"] == topic]),
        }
        for topic, val in low_sentiment_topics.items()
    ]

    return {"positive": positive_themes, "negative": negative_themes}


def generate_word_clouds(real_df: pd.DataFrame):
    """Generate word cloud(s) from customer feedback text.

    If Timestamp column exists: generates one word cloud per time period (quarterly).
    Otherwise: generates a single aggregate word cloud.

    Returns:
        Dict with 'clouds' (list of (period_label, image_bytes) tuples) and 'has_time' (bool).
    """
    if real_df is None or real_df.empty:
        return {"clouds": [], "has_time": False}

    # Find the text column(s) to use
    text_col = None
    for col in ["Interaction_Payload", "Improvement_Comments", "feedback_text"]:
        if col in real_df.columns:
            text_col = col
            break

    if text_col is None:
        return {"clouds": [], "has_time": False}

    # Define extended stopwords for B2B context
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(
        [
            "will",
            "would",
            "could",
            "should",
            "also",
            "really",
            "just",
            "us",
            "our",
            "we",
            "they",
            "their",
            "them",
            "you",
            "your",
            "one",
            "two",
            "three",
            "first",
            "second",
            "new",
            "like",
            "make",
            "made",
            "get",
            "got",
            "know",
            "think",
            "see",
            "use",
            "used",
            "using",
            "need",
            "want",
            "well",
            "even",
            "much",
            "company",
            "companies",
            "client",
            "clients",
            "customer",
            "customers",
            "team",
            "teams",
            "product",
            "products",
            "service",
            "services",
        ]
    )

    def clean_text(text_series):
        """Preprocess text for word cloud generation."""
        combined = " ".join(text_series.dropna().astype(str))
        # Remove URLs, emails, special characters
        combined = re.sub(r"http\S+|www\.\S+", "", combined)
        combined = re.sub(r"\S+@\S+", "", combined)
        combined = re.sub(r"[^a-zA-Z\s]", " ", combined)
        # Lowercase and remove extra whitespace
        combined = combined.lower()
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined

    def create_wordcloud_image(text, title=""):
        """Generate a word cloud image and return as bytes."""
        if not text or len(text.split()) < 10:
            return None

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=custom_stopwords,
            max_words=100,
            colormap="viridis",
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=120,
            relative_scaling=0.5,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50", pad=10)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue()

    clouds = []
    has_time = False

    # Check if we have timestamp data for time-based word clouds
    if "Timestamp" in real_df.columns and pd.notna(real_df["Timestamp"]).any():
        has_time = True
        df_time = real_df.copy()
        df_time["Timestamp"] = pd.to_datetime(df_time["Timestamp"], errors="coerce")
        df_time = df_time.dropna(subset=["Timestamp"])

        if not df_time.empty:
            # Create quarterly periods
            df_time["Quarter"] = df_time["Timestamp"].dt.to_period("Q").astype(str)

            for quarter in sorted(df_time["Quarter"].unique()):
                quarter_data = df_time[df_time["Quarter"] == quarter]
                text = clean_text(quarter_data[text_col])
                img_bytes = create_wordcloud_image(
                    text, f"Q{quarter[-1]} {quarter[:4]}"
                )
                if img_bytes:
                    clouds.append((quarter, img_bytes))

    # If no time-based clouds or not enough data, create aggregate
    if not clouds:
        has_time = False
        text = clean_text(real_df[text_col])
        img_bytes = create_wordcloud_image(text, "All Feedback")
        if img_bytes:
            clouds.append(("All", img_bytes))

    return {"clouds": clouds, "has_time": has_time}


def get_temporal_summary_stats(real_df: pd.DataFrame, granularity: str = "Monthly"):
    """Generate summary statistics for temporal analysis.

    Args:
        real_df: The main dataframe with feedback data
        granularity: Time granularity ('Daily', 'Weekly', 'Monthly', 'Quarterly')

    Returns:
        Dict with summary statistics by time period
    """
    if real_df is None or real_df.empty or "Timestamp" not in real_df.columns:
        return {}

    df = real_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        return {}

    # Create period column
    if granularity == "Daily":
        df["Period"] = df["Timestamp"].dt.date
    elif granularity == "Weekly":
        df["Period"] = df["Timestamp"].dt.to_period("W").dt.start_time
    elif granularity == "Monthly":
        df["Period"] = df["Timestamp"].dt.to_period("M").dt.start_time
    else:  # Quarterly
        df["Period"] = df["Timestamp"].dt.to_period("Q").dt.start_time

    # Aggregate by period
    period_stats = (
        df.groupby("Period")
        .agg(
            {
                "NPS_Score": ["mean", "std", "count"],
                "CSAT_Score": ["mean", "std"],
                "sentiment_polarity": "mean"
                if "sentiment_polarity" in df.columns
                else lambda x: None,
                "Account_Name": "nunique",
            }
        )
        .reset_index()
    )

    return period_stats.to_dict("records")


def detect_trend_changes(
    real_df: pd.DataFrame,
    metric: str = "NPS_Score",
    granularity: str = "Monthly",
    threshold: float = 1.0,
):
    """Detect significant trend changes in a metric over time.

    Args:
        real_df: The main dataframe
        metric: The metric to analyze ('NPS_Score', 'CSAT_Score', etc.)
        granularity: Time granularity
        threshold: Minimum change to be considered significant

    Returns:
        List of dicts with trend change information
    """
    if real_df is None or real_df.empty or metric not in real_df.columns:
        return []

    df = real_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        return []

    # Create period column
    if granularity == "Daily":
        df["Period"] = df["Timestamp"].dt.date
    elif granularity == "Weekly":
        df["Period"] = df["Timestamp"].dt.to_period("W").dt.start_time
    elif granularity == "Monthly":
        df["Period"] = df["Timestamp"].dt.to_period("M").dt.start_time
    else:  # Quarterly
        df["Period"] = df["Timestamp"].dt.to_period("Q").dt.start_time

    # Calculate metric by period
    metric_by_period = df.groupby("Period")[metric].mean().sort_index()

    # Calculate period-over-period changes
    changes = []
    for i in range(1, len(metric_by_period)):
        prev_val = metric_by_period.iloc[i - 1]
        curr_val = metric_by_period.iloc[i]
        change = curr_val - prev_val

        if abs(change) >= threshold:
            changes.append(
                {
                    "period": metric_by_period.index[i],
                    "previous_value": round(prev_val, 2),
                    "current_value": round(curr_val, 2),
                    "change": round(change, 2),
                    "pct_change": round(
                        (change / prev_val * 100) if prev_val != 0 else 0, 2
                    ),
                    "direction": "increase" if change > 0 else "decrease",
                }
            )

    return changes


def calculate_temporal_deltas(
    real_df: pd.DataFrame, selected_date: str = None
) -> Dict[str, Any]:
    """Calculate period-over-period deltas for key metrics.

    Args:
        real_df: The main dataframe with feedback data
        selected_date: Optional specific date to compare against previous period

    Returns:
        Dict with delta values and percentages for NPS, CSAT, health score, and feedback count
    """
    if real_df is None or real_df.empty or "Timestamp" not in real_df.columns:
        return {
            "nps_delta": None,
            "csat_delta": None,
            "health_delta": None,
            "feedback_delta": None,
            "feedback_delta_pct": None,
        }

    df = real_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        return {
            "nps_delta": None,
            "csat_delta": None,
            "health_delta": None,
            "feedback_delta": None,
            "feedback_delta_pct": None,
        }

    # Sort by timestamp
    df = df.sort_values("Timestamp")

    # If a specific date is selected, compare it to the previous date
    if selected_date and selected_date != "Overall":
        dates_series = df["Timestamp"].dt.date.astype(str)
        unique_dates = sorted(dates_series.unique().tolist())

        if selected_date in unique_dates:
            current_idx = unique_dates.index(selected_date)

            # Get current period data
            current_data = df[dates_series == selected_date]

            # Get previous period data if available
            if current_idx > 0:
                prev_date = unique_dates[current_idx - 1]
                prev_data = df[dates_series == prev_date]
            else:
                # No previous period available
                return {
                    "nps_delta": None,
                    "csat_delta": None,
                    "health_delta": None,
                    "feedback_delta": None,
                    "feedback_delta_pct": None,
                }
        else:
            return {
                "nps_delta": None,
                "csat_delta": None,
                "health_delta": None,
                "feedback_delta": None,
                "feedback_delta_pct": None,
            }
    else:
        # Compare last month vs previous month
        df["Period"] = df["Timestamp"].dt.to_period("M")
        periods = sorted(df["Period"].unique())

        if len(periods) < 2:
            # Not enough data for comparison
            return {
                "nps_delta": None,
                "csat_delta": None,
                "health_delta": None,
                "feedback_delta": None,
                "feedback_delta_pct": None,
            }

        current_period = periods[-1]
        prev_period = periods[-2]

        current_data = df[df["Period"] == current_period]
        prev_data = df[df["Period"] == prev_period]

    # Calculate current metrics
    current_sentiment = get_sentiment_insights(current_data)
    current_nps = current_sentiment.get("nps", 0)
    current_csat = (
        current_data["CSAT_Score"].mean() if "CSAT_Score" in current_data.columns else 0
    )
    current_health = calculate_overall_health_score(current_nps, current_csat)
    current_count = len(current_data)

    # Calculate previous metrics
    prev_sentiment = get_sentiment_insights(prev_data)
    prev_nps = prev_sentiment.get("nps", 0)
    prev_csat = (
        prev_data["CSAT_Score"].mean() if "CSAT_Score" in prev_data.columns else 0
    )
    prev_health = calculate_overall_health_score(prev_nps, prev_csat)
    prev_count = len(prev_data)

    # Calculate deltas
    nps_delta = round(current_nps - prev_nps, 1) if prev_nps != 0 else None
    csat_delta = round(current_csat - prev_csat, 1) if prev_csat != 0 else None
    health_delta = round(current_health - prev_health, 1) if prev_health != 0 else None

    feedback_delta = current_count - prev_count
    feedback_delta_pct = (
        round((feedback_delta / prev_count * 100), 1) if prev_count > 0 else None
    )

    return {
        "nps_delta": nps_delta,
        "csat_delta": csat_delta,
        "health_delta": health_delta,
        "feedback_delta": feedback_delta,
        "feedback_delta_pct": feedback_delta_pct,
    }
