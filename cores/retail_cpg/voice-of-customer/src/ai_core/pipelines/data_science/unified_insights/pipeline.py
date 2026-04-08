from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_unified_interaction_record, create_customer_aspect_profiles

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_unified_interaction_record,
                inputs=[
                    "sentiment_predictions",
                    "aspect_sentiment_predictions",
                    "topic_assignments",
                    "emotion_predictions"
                ],
                outputs="unified_interaction_records",
                name="create_unified_interaction_record_node",
            ),
            node(
                func=create_customer_aspect_profiles,
                inputs="aspect_sentiment_predictions",
                outputs="customer_aspect_profiles",
                name="create_customer_aspect_profiles_node",
            ),
        ]
    )
