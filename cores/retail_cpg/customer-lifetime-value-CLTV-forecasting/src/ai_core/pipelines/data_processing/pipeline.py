from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_data,
    transform_transaction_data,
    map_to_skeleton,
    mask_pii,
    prepare_sequences,
    register_rfm_features_to_feast,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=map_to_skeleton,
                inputs=["raw_data", "params:data_processing"],
                outputs="skeleton_data",
                name="map_to_skeleton_node",
            ),
            node(
                func=mask_pii,
                inputs=["skeleton_data", "params:data_processing"],
                outputs="masked_data",
                name="mask_pii_node",
            ),
            node(
                func=clean_data,
                inputs=["masked_data", "params:data_processing"],
                outputs="cleaned_customer_data",
                name="clean_data_node",
            ),
            node(
                func=transform_transaction_data,
                inputs=["cleaned_customer_data", "params:data_processing"],
                outputs="processed_data",
                name="transform_transaction_data_node",
            ),
            node(
                func=prepare_sequences,
                inputs=["cleaned_customer_data", "params:data_processing"],
                outputs="engagement_sequences",
                name="prepare_sequences_node",
            ),
            node(
                func=register_rfm_features_to_feast,
                inputs=["processed_data", "params:feast"],
                outputs="feast_registration_status",
                name="register_rfm_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ]
    )
