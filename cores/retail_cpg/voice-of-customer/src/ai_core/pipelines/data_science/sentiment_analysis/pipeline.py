from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_sentiment_model,
    analyze_sentiment,
    create_sentiment_summary,
    create_customer_sentiment_profiles,
    create_customer_sentiment_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the sentiment analysis pipeline"""
    return pipeline(
        [
            node(
                func=load_sentiment_model,
                inputs="params:sentiment_analysis.model",
                outputs="sentiment_model",
                name="load_sentiment_model_node",
                tags=["model_loading", "sentiment"]
            ),
            node(
                func=analyze_sentiment,
                inputs=[
                    "review_features",
                    "sentiment_model",
                    "params:sentiment_analysis.model"
                ],
                outputs="sentiment_predictions",
                name="analyze_sentiment_node",
                tags=["sentiment_analysis", "inference"]
            ),
            node(
                func=create_sentiment_summary,
                inputs="sentiment_predictions",
                outputs="sentiment_summary",
                name="create_sentiment_summary_node",
                tags=["sentiment_analysis", "reporting"]
            ),
            # Customer-level sentiment profiling
            node(
                func=create_customer_sentiment_profiles,
                inputs=[
                    "sentiment_predictions",
                    "params:customer_analysis"
                ],
                outputs="customer_sentiment_profiles",
                name="create_customer_sentiment_profiles_node",
                tags=["sentiment_analysis", "customer_profiling"]
            ),
            node(
                func=create_customer_sentiment_summary,
                inputs=[
                    "customer_sentiment_profiles",
                    "params:customer_analysis"
                ],
                outputs="customer_sentiment_summary",
                name="create_customer_sentiment_summary_node",
                tags=["sentiment_analysis", "customer_profiling", "reporting"]
            ),
        ],
        tags=["sentiment_pipeline"]
    )
