from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    check_sentiment_drift,
    check_topic_drift,
    generate_alerts,
    detect_high_urgency_alerts
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the monitoring pipeline"""
    return pipeline(
        [
            node(
                func=check_sentiment_drift,
                inputs=["sentiment_summary", "params:monitoring"],
                outputs="sentiment_drift_metrics",
                name="check_sentiment_drift_node",
                tags=["monitoring", "drift_detection"]
            ),
            node(
                func=check_topic_drift,
                inputs=["topic_summary", "params:monitoring"],
                outputs="topic_drift_metrics",
                name="check_topic_drift_node",
                tags=["monitoring", "drift_detection"]
            ),
            node(
                func=generate_alerts,
                inputs=[
                    "sentiment_drift_metrics",
                    "topic_drift_metrics",
                    "params:monitoring"
                ],
                outputs="monitoring_alerts",
                name="generate_alerts_node",
                tags=["monitoring", "alerting"]
            ),
            node(
                func=detect_high_urgency_alerts,
                inputs="unified_interaction_records",
                outputs="high_urgency_alerts",
                name="detect_high_urgency_alerts_node",
                tags=["monitoring", "alerting"]
            ),
        ],
        tags=["monitoring_pipeline"]
    )
