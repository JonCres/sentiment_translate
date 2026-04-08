from kedro.pipeline import Pipeline, node, pipeline
from .nodes import analyze_text_emotions

def create_mer_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=analyze_text_emotions,
                inputs=["reviews_cleaned", "params:mer.model"],
                outputs="emotion_predictions",
                name="analyze_text_emotions_node",
            ),
        ]
    )
