import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


def analyze_correlations(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze correlations between NLP metrics and business scores (NPS/CSAT).

    Args:
        data: Enriched DataFrame with sentiments, emotions, topics, and ratings.
        parameters: Analysis parameters.

    Returns:
        Dictionary containing correlation matrices and key insights.
    """
    logger.info(
        "Starting correlation analysis between NLP features and business metrics..."
    )

    # Identify relevant columns
    sentiment_cols = [c for c in data.columns if c.startswith("sent_")]
    emotion_cols = [c for c in data.columns if c.startswith("emo_")]

    # For topics, we one-hot encode to calculate correlations
    topic_cols = []
    if "Topic_Name" in data.columns:
        topic_dummies = pd.get_dummies(data["Topic_Name"], prefix="topic")
        topic_cols = topic_dummies.columns.tolist()
        data = pd.concat([data, topic_dummies], axis=1)

    # Business targets
    targets = []
    for t in ["NPS_Score", "CSAT_Score", "nps_norm", "csat_norm"]:
        if t in data.columns:
            targets.append(t)

    if not targets:
        logger.warning("No business targets (NPS/CSAT) found for analysis.")
        return {"status": "skipped", "reason": "no_targets"}

    feature_cols = sentiment_cols + emotion_cols + topic_cols

    if not feature_cols:
        logger.warning(
            "No NLP features (sentiments, emotions, topics) found for analysis."
        )
        return {"status": "skipped", "reason": "no_features"}

    # Calculate correlations
    correlation_results = {}

    for target in targets:
        corrs = (
            data[feature_cols + [target]]
            .corr()[target]
            .drop(target)
            .sort_values(ascending=False)
        )
        correlation_results[target] = corrs.to_dict()

    # Calculate feature importance/drivers
    # We can also do a simple linear regression to get coefficients if wanted,
    # but correlation is what was requested.

    # Identify top positive and negative drivers for NPS
    insights = {}
    if "NPS_Score" in correlation_results:
        nps_corrs = pd.Series(correlation_results["NPS_Score"])
        insights["top_positive_drivers"] = nps_corrs.head(5).to_dict()
        insights["top_negative_drivers"] = nps_corrs.tail(5).to_dict()

    return {
        "correlations": correlation_results,
        "insights": insights,
        "status": "success",
    }
