from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    map_to_skeleton,
    validate_data,
    detect_languages,
    translate_feedback,
    clean_feedback_data,
    engineer_voc_features,
    register_features_to_feast,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=map_to_skeleton,
                inputs=["raw_feedback", "params:data_processing.skeleton_mapping"],
                outputs="skeleton_data",
                name="map_to_skeleton_node",
            ),
            node(
                func=validate_data,
                inputs=["skeleton_data", "params:data_processing"],
                outputs="validated_feedback",
                name="validate_feedback_node",
            ),
            node(
                func=detect_languages,
                inputs=["validated_feedback", "params:data_processing"],
                outputs="feedback_with_language",
                name="detect_languages_node",
            ),
            node(
                func=translate_feedback,
                inputs=["feedback_with_language", "params:data_processing"],
                outputs="translated_feedback",
                name="translate_feedback_node",
            ),
            node(
                func=clean_feedback_data,
                inputs="translated_feedback",
                outputs="clean_feedback",
                name="clean_feedback_node",
            ),
            node(
                func=engineer_voc_features,
                inputs=["clean_feedback", "params:data_processing"],
                outputs="feedback_features",
                name="engineer_voc_features_node",
            ),
            node(
                func=register_features_to_feast,
                inputs=["feedback_features", "params:feast"],
                outputs="feast_registration_status",
                name="register_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ]
    )
