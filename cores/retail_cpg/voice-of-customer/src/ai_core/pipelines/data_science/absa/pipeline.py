"""
ABSA (Aspect-Based Sentiment Analysis) Pipeline

Orchestrates aspect extraction and per-aspect sentiment analysis.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    extract_aspects,
    load_absa_model,
    analyze_aspect_sentiment,
    create_aspect_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ABSA pipeline."""
    return pipeline(
        [
            node(
                func=load_absa_model,
                inputs="params:absa.model",
                outputs="absa_model",
                name="load_absa_model_node",
                tags=["absa", "model_loading"]
            ),
            node(
                func=extract_aspects,
                inputs=[
                    "review_features",
                    "params:absa.aspects"
                ],
                outputs="reviews_with_aspects",
                name="extract_aspects_node",
                tags=["absa", "aspect_extraction"]
            ),
            node(
                func=analyze_aspect_sentiment,
                inputs=[
                    "reviews_with_aspects",
                    "absa_model",
                    "params:absa.model"
                ],
                outputs="aspect_sentiment_predictions",
                name="analyze_aspect_sentiment_node",
                tags=["absa", "sentiment_analysis"]
            ),
            node(
                func=create_aspect_summary,
                inputs=[
                    "aspect_sentiment_predictions",
                    "params:absa"
                ],
                outputs="aspect_sentiment_summary",
                name="create_aspect_summary_node",
                tags=["absa", "reporting"]
            ),
        ],
        tags=["absa_pipeline"]
    )
