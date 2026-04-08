from kedro.pipeline import Pipeline, node
from .nodes import analyze_sentiment, generate_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=analyze_sentiment,
            inputs="translated_reviews",
            outputs="sentiment_reviews",
            name="sentiment_node"
        ),
        node(
            func=generate_report,
            inputs="sentiment_reviews",
            outputs=None,
            name="sentiment_report_node"
        )
    ])