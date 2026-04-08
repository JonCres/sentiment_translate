from kedro.pipeline import Pipeline, node, pipeline
from .nodes import analyze_correlations


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=analyze_correlations,
                inputs=["final_enriched_feedback", "params:modeling"],
                outputs="nlp_business_correlations",
                name="analyze_correlations_node",
                tags=["analysis", "correlations"],
            ),
        ]
    )
