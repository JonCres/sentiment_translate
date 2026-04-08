from kedro.pipeline import Pipeline, node
from .nodes import validate_translation, generate_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=validate_translation,
            inputs=["translated_reviews", "params"],
            outputs="validated_reviews",
            name="validation_node"
        ),
        node(
            func=generate_report,
            inputs="validated_reviews",
            outputs=None,
            name="validation_report_node"
        )
    ])