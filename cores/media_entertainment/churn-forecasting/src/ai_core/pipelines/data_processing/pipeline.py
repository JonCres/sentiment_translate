from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_transaction_skeleton,
    clean_skeleton_data,
    transform_transaction_data,
    create_subscriptions_skeleton,
    create_engagement_skeleton,
    create_survival_data,
    create_feature_store,
    create_tensor_sequences,
    register_churn_features_to_feast,
    create_qoe_skeleton,
    create_social_graph_skeleton,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Backbone (Transactions) ---
            node(
                func=create_transaction_skeleton,
                inputs=["raw_data", "params:skeleton"],
                outputs="transactions_skeleton",
                name="create_transaction_skeleton_node",
            ),
            node(
                func=clean_skeleton_data,
                inputs="transactions_skeleton",
                outputs="cleaned_customer_data",
                name="clean_skeleton_data_node",
            ),
            node(
                func=transform_transaction_data,
                inputs=["cleaned_customer_data", "params:data_processing"],
                outputs="processed_data",
                name="transform_transaction_data_node",
            ),
            # --- Contractual (Subscriptions) ---
            node(
                func=create_subscriptions_skeleton,
                inputs=["raw_subscriptions", "params:skeleton"],
                outputs="subscriptions_skeleton",
                name="create_subscriptions_skeleton_node",
            ),
            node(
                func=create_survival_data,
                inputs="subscriptions_skeleton",
                outputs="survival_data",
                name="create_survival_data_node",
            ),
            # --- Behavioral (Engagement) ---
            node(
                func=create_engagement_skeleton,
                inputs=["raw_engagement", "params:skeleton"],
                outputs="engagement_skeleton",
                name="create_engagement_skeleton_node",
            ),
            node(
                func=create_qoe_skeleton,
                inputs=["raw_qoe", "params:skeleton"],
                outputs="qoe_skeleton",
                name="create_qoe_skeleton_node",
            ),
             node(
                func=create_social_graph_skeleton,
                inputs=["raw_social_graph", "params:skeleton"],
                outputs="social_graph_skeleton",
                name="create_social_graph_skeleton_node",
            ),
            # --- Refinement (Feature Store) ---
            # Joins Backbone + Contractual + Behavioral data, and tensor features
            node(
                func=create_feature_store,
                inputs={
                    "transactions": "processed_data",
                    "subscriptions": "subscriptions_skeleton",
                    "engagement": "engagement_skeleton",
                    "qoe": "qoe_skeleton",
                    "social": "social_graph_skeleton",
                    "tensor_features": "tensor_sequences",
                },
                outputs="feature_store",
                name="create_feature_store_node",
            ),
            # --- Deep Learning Data Prep ---
            node(
                func=create_tensor_sequences,
                inputs=["engagement_skeleton", "processed_data", "parameters"],
                outputs=["tensor_sequences", "tensor_labels"],
                name="create_tensor_sequences_node",
            ),
            # --- Feast Registration ---
            node(
                func=register_churn_features_to_feast,
                inputs=[
                    "processed_data",
                    "feature_store",
                    "params:feast",
                ],
                outputs="feast_registration_status",
                name="register_churn_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ]
    )
