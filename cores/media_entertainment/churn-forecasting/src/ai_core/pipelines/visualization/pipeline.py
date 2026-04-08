from kedro.pipeline import Pipeline, node, pipeline

from .nodes import plot_shap_summary, evaluate_models, plot_model_comparison


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
                func=evaluate_models,
                inputs=[
                    "test_data",
                    "parameters",
                    "xgboost_model",
                    "random_forest_model",
                    "lightgbm_model",
                    "catboost_model",
                    "ensemble_model",
                ],
                outputs="model_metrics",
                name="evaluate_models_node",
            ),
            node(
                func=plot_model_comparison,
                inputs=[
                    "test_data",
                    "parameters",
                    "xgboost_model",
                    "random_forest_model",
                    "lightgbm_model",
                    "catboost_model",
                    "ensemble_model",
                ],
                outputs="model_comparison_plot",
                name="plot_model_comparison_node",
            ),
        ]
    )
