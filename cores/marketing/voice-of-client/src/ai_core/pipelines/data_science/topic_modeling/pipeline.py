from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_topic_model, assign_topics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_topic_model,
                inputs=[
                    "feedback_with_churn",
                    "params:modeling.topic",
                ],
                outputs="topic_model",
                name="train_topic_model_node",
                tags=["topic_modeling"],
            ),
            node(
                func=assign_topics,
                inputs=["feedback_with_churn", "topic_model", "params:modeling.topic"],
                outputs="final_enriched_feedback",
                name="assign_topics_node",
                tags=["topic_modeling"],
            ),
        ]
    )
