from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    validate_data,
    clean_data,
    engineer_features,
    register_features_to_feast,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data,
                inputs=["raw_data", "params:data_pipeline.validation"],
                outputs="validated_data",
                name="validate_data_node",
            ),
            node(
                func=clean_data,
                inputs="validated_data",
                outputs="clean_data",
                name="clean_data_node",
            ),
            node(
                func=engineer_features,
                inputs=["clean_data", "params:data_pipeline.features"],
                outputs="features_data",
                name="engineer_features_node",
            ),
            node(
                func=register_features_to_feast,
                inputs=["features_data", "params:feast"],
                outputs="feast_registration_status",
                name="register_features_to_feast_node",
                tags=["feast", "feature_store"],
            ),
        ]
    )
