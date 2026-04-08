from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    plot_shap_summary,
    evaluate_models,
    plot_model_comparison,
    plot_survival_curves,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_shap_summary,
                inputs="shap_explanation",
                outputs="shap_summary_plot",
                name="plot_shap_summary_node",
            ),
            node(
                func=plot_survival_curves,
                inputs=["cox_ph_model", "survival_data_prepared", "parameters"],
                outputs="survival_curves_plot",
                name="plot_survival_curves_node",
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "survival_data_prepared",  # used as test_data
                    "parameters",
                    "cox_ph_model",
                    "rsf_model",
                    "deepsurv_model",
                    "nmtlr_model",
                ],
                outputs="model_metrics",
                name="evaluate_models_node",
            ),
            node(
                func=plot_model_comparison,
                inputs="model_metrics",
                outputs="model_comparison_plot",
                name="plot_model_comparison_node",
            ),
        ]
    )
