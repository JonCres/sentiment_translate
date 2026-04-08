from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    predict_churn,
    calculate_business_metrics,
    prepare_model_input,
    evaluate_survival_model,
    train_coxph_model,
    train_rsf_model,
    explain_survival_model_shap,
    explain_survival_model_lime,
    evaluate_explanation_faithfulness,
    evaluate_explanation_consistency,
)
from .dl_nodes import train_deepsurv_model, train_nmtlr_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Survival Data Preparation ---
            node(
                func=prepare_model_input,
                inputs=["feature_store", "survival_data_prepared"],
                outputs="survival_train_data",
                name="prepare_model_input_node",
            ),
            # --- Statistical Survival Model (CPH) ---
            node(
                func=train_coxph_model,
                inputs=["survival_train_data", "parameters"],
                outputs="cox_ph_model",
                name="train_coxph_node",
            ),
            # --- ML Survival Model (Random Survival Forest) ---
            node(
                func=train_rsf_model,
                inputs=["survival_train_data", "parameters"],
                outputs="rsf_model",
                name="train_rsf_node",
            ),
            # --- Deep Learning Survival Models ---
            node(
                func=train_deepsurv_model,
                inputs=[
                    "survival_train_data",
                    "params:modeling.survival_analysis.deepsurv",
                ],
                outputs="deepsurv_model",
                name="train_deepsurv_node",
            ),
            node(
                func=train_nmtlr_model,
                inputs=[
                    "survival_data_prepared", # NMTLR might need simpler dict format or update nmtlr node too? 
                    # For safety, let's leave NMTLR as is or disable it if it breaks. 
                    # Assuming NMTLR node wrapper handles it or it wasn't updated. 
                    # dl_nodes.py train_nmtlr_model still takes Dict[str, pd.DataFrame]. 
                    # We should probably skip wiring it to `survival_train_data` unless we updated it.
                    "params:modeling.survival_analysis.nmtlr",
                ],
                outputs="nmtlr_model",
                name="train_nmtlr_node",
            ),
            # --- Evaluation ---
            node(
                func=evaluate_survival_model,
                inputs=["deepsurv_model", "survival_train_data"],
                outputs="model_metrics_dict",
                name="evaluate_deepsurv_node",
            ),
            # --- Explainability (XAI) ---
            node(
                func=explain_survival_model_shap,
                inputs=["deepsurv_model", "survival_train_data", "parameters"], # Use DeepSurv for XAI
                outputs="shap_explanation",
                name="explain_deepsurv_shap_node",
            ),
            node(
                func=explain_survival_model_lime,
                inputs=["deepsurv_model", "survival_train_data", "parameters"],
                outputs="lime_explanation",
                name="explain_deepsurv_lime_node",
            ),
            # --- XAI Evaluation ---
            node(
                func=evaluate_explanation_faithfulness,
                inputs=[
                    "deepsurv_model",
                    "survival_train_data",
                    "shap_explanation",
                    "parameters",
                ],
                outputs="faithfulness_score",
                name="evaluate_shap_faithfulness_node",
            ),
            node(
                func=evaluate_explanation_consistency,
                inputs=["shap_explanation", "survival_train_data"],
                outputs="consistency_score",
                name="evaluate_shap_consistency_node",
            ),
            # --- Inference & Business Metrics ---
            node(
                func=predict_churn,
                inputs=[
                    "deepsurv_model",  # Use DeepSurv
                    "feature_store",  # Use feature store directly
                ],
                outputs="raw_churn_predictions",
                name="predict_churn_node",
            ),
            node(
                func=calculate_business_metrics,
                inputs=["raw_churn_predictions", "feature_store"],
                outputs="churn_predictions",
                name="calculate_business_metrics_node",
            ),
        ]
    )
