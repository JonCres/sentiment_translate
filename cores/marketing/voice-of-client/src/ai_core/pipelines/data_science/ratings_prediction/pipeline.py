from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict_nps_and_csat


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict_nps_and_csat,
                inputs=["feedback_with_emotions", "params:modeling"],
                outputs=["feedback_with_ratings", "ratings_models_metrics"],
                name="predict_ratings_node",
                tags=["ratings"],
            ),
        ]
    )
