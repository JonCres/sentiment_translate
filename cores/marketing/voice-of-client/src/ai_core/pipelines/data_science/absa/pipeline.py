from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_absa_inference


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_absa_inference,
                inputs=["feedback_features", "params:modeling"],
                outputs="feedback_with_sentiment",
                name="absa_inference_node",
                tags=["absa"],
            ),
        ]
    )
