from kedro.pipeline import Pipeline, node
from .nodes import translate_text, generate_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline ([
        node(
            func=translate_text,
            inputs="raw_reviews",
            outputs="translated_reviews",
            name="translate_node"
            ),
        node(
            func=generate_report,
            inputs="translated_reviews",
            outputs=None,
            name="translation_report_node"
        )
        ])

        # node(
         #   func=translate_text,
         #   inputs="speech_reviews",  # 👈 importante
         #   outputs="translated_reviews",
         #   name="translate_node"
        #)