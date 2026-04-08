from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_transaction_skeleton,
    clean_skeleton_data,
    transform_transaction_data,
    create_subscriptions_skeleton,
    create_engagement_skeleton,
    create_survival_data,
    create_counting_process_data,
    create_feature_store,
    create_tensor_sequences,
    register_churn_features_to_feast,
    anonymize_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Backbone (Transactions) ---
            node(
                func=anonymize_data,
                inputs=["raw_data", "params:skeleton"],
                outputs="anonymized_raw_data",
                name="anonymize_raw_data_node",
            ),
            node(
                func=create_transaction_skeleton,
                inputs=["anonymized_raw_data", "params:skeleton"],
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
                inputs=["subscriptions_skeleton", "params:skeleton"],
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
                func=create_counting_process_data,
                inputs=["survival_data", "engagement_skeleton", "params:skeleton"],
                outputs="survival_data_prepared",
                name="create_counting_process_data_node",
            ),
            # --- Refinement (Feature Store) ---
            # Joins Backbone + Contractual + Behavioral
            node(
                func=create_feature_store,
                inputs=[
                    "transactions_skeleton",
                    "subscriptions_skeleton",
                    "engagement_skeleton",
                ],
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
                    "survival_data",
                    "feature_store",
                    "params:feast",
                ],
                outputs="feast_registration_status",
                name="register_churn_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ]
    )
