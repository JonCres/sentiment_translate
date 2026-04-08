from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_topic_model,
    predict_topics,
    create_topic_summary,
    create_customer_topic_profiles,
    create_customer_topic_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the topic modeling pipeline"""
    return pipeline(
        [
            node(
                func=train_topic_model,
                inputs=[
                    "review_features",
                    "params:topic_modeling.model",
                    "params:topic_modeling.reduction",
                    "params:topic_modeling.clustering"
                ],
                outputs="topic_model",
                name="train_topic_model_node",
                tags=["topic_modeling", "training"]
            ),
            node(
                func=predict_topics,
                inputs=["review_features", "topic_model"],
                outputs="topic_assignments",
                name="predict_topics_node",
                tags=["topic_modeling", "inference"]
            ),
            node(
                func=create_topic_summary,
                inputs=["topic_assignments", "topic_model"],
                outputs="topic_summary",
                name="create_topic_summary_node",
                tags=["topic_modeling", "reporting"]
            ),
            # Customer-level topic profiling
            node(
                func=create_customer_topic_profiles,
                inputs=[
                    "topic_assignments",
                    "topic_model",
                    "params:customer_analysis"
                ],
                outputs="customer_topic_profiles",
                name="create_customer_topic_profiles_node",
                tags=["topic_modeling", "customer_profiling"]
            ),
            node(
                func=create_customer_topic_summary,
                inputs=[
                    "customer_topic_profiles",
                    "params:customer_analysis"
                ],
                outputs="customer_topic_summary",
                name="create_customer_topic_summary_node",
                tags=["topic_modeling", "customer_profiling", "reporting"]
            ),
        ],
        tags=["topic_pipeline"]
    )
