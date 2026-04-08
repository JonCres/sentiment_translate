from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_context_aware_scoring


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_context_aware_scoring,
                inputs=["feedback_with_emotions", "params:modeling"],
                outputs="feedback_with_context_sentiment",
                name="context_aware_scoring_node",
                tags=["sentiment_scoring"],
            ),
        ]
    )
