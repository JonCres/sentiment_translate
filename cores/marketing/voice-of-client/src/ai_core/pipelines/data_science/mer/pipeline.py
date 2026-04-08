from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_mer_inference


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_mer_inference,
                inputs=["feedback_with_sentiment", "params:modeling"],
                outputs="feedback_with_emotions",
                name="mer_inference_node",
                tags=["mer"],
            ),
        ]
    )
