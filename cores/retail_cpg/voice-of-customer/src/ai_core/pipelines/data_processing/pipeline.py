from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_reviews_from_csv,
    map_to_skeleton,
    translate_reviews,
    validate_reviews,
    clean_reviews,
    engineer_features,
    register_review_features_to_feast,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline"""
    return pipeline(
        [
            node(
                func=load_reviews_from_csv,
                inputs=["amazon_reviews_raw", "params:data_processing.source"],
                outputs="reviews_loaded",
                name="load_reviews_node",
                tags=["data_load", "data_processing"],
            ),
            node(
                func=map_to_skeleton,
                inputs=["reviews_loaded", "params:skeleton_mapping"],
                outputs="reviews_skeleton",
                name="map_to_skeleton_node",
                tags=["skeleton_mapping", "data_processing"],
            ),
            node(
                func=translate_reviews,
                inputs=["reviews_skeleton", "params:data_processing.translation"],
                outputs="reviews_translated",
                name="translate_reviews_node",
                tags=["translation", "data_processing"],
            ),
            node(
                func=validate_reviews,
                inputs=["reviews_translated", "params:data_processing.validation"],
                outputs="reviews_validated",
                name="validate_reviews_node",
                tags=["data_validation", "data_processing"],
            ),
            node(
                func=clean_reviews,
                inputs=["reviews_validated", "params:data_processing.cleaning"],
                outputs="reviews_cleaned",
                name="clean_reviews_node",
                tags=["data_cleaning", "data_processing"],
            ),
            node(
                func=engineer_features,
                inputs=["reviews_cleaned", "params:data_processing.features"],
                outputs="review_features",
                name="engineer_features_node",
                tags=["feature_engineering", "data_processing"],
            ),
            node(
                func=register_review_features_to_feast,
                inputs=["review_features", "params:feast"],
                outputs="feast_registration_status",
                name="register_review_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ],
        tags=["data_pipeline"],
    )
