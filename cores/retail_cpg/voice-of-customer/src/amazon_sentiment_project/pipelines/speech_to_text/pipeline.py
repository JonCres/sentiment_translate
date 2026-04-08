from kedro.pipeline import Pipeline, node
from .nodes import transcribe_audio

def create_pipeline(**kwargs)  -> Pipeline:
    return Pipeline([
        node(
            func=transcribe_audio,
            inputs="raw_reviews",
            outputs="speech_reviews",
            name="speech_to_text_node"
        )
    ])