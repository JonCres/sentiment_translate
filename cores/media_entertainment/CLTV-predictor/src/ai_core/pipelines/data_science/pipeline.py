from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    predict_cltv,
    calculate_business_metrics,
    prepare_survival_data,
    train_coxph_model,
    train_rsf_model,
    train_bg_nbd_model,
    train_bg_nbd_model,
    train_gamma_gamma_model,
    train_tweedie_model,
    explain_survival_model_shap,
    explain_survival_model_lime,
    evaluate_explanation_faithfulness,
    evaluate_explanation_consistency,
)
from .dl_nodes import (
    train_deepsurv_model,
    train_nmtlr_model,
    train_sequential_cltv_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Survival Data Preparation ---
            node(
                func=prepare_survival_data,
                inputs=["feature_store", "parameters"],
                outputs="survival_data_prepared",
                name="prepare_survival_data_node",
            ),
            # --- Statistical Survival Model (CPH) ---
            node(
                func=train_coxph_model,
                inputs=["survival_data_prepared", "parameters"],
                outputs="cox_ph_model",
                name="train_coxph_node",
            ),
            # --- ML Survival Model (Random Survival Forest) ---
            node(
                func=train_rsf_model,
                inputs=["survival_data_prepared", "parameters"],
                outputs="rsf_model",
                name="train_rsf_node",
            ),
            # --- Deep Learning Survival Models ---
            node(
                func=train_deepsurv_model,
                inputs=[
                    "survival_data_prepared",
                    "params:modeling.survival_analysis.deepsurv",
                ],
                outputs="deepsurv_model",
                name="train_deepsurv_node",
            ),
            node(
                func=train_nmtlr_model,
                inputs=[
                    "survival_data_prepared",
                    "params:modeling.survival_analysis.nmtlr",
                ],
                outputs="nmtlr_model",
                name="train_nmtlr_node",
            ),
            # --- Explainability (XAI) ---
            node(
                func=explain_survival_model_shap,
                inputs=["cox_ph_model", "survival_data_prepared", "parameters"],
                outputs="shap_explanation",
                name="explain_coxph_shap_node",
            ),
            node(
                func=explain_survival_model_lime,
                inputs=["cox_ph_model", "survival_data_prepared", "parameters"],
                outputs="lime_explanation",
                name="explain_coxph_lime_node",
            ),
            # --- XAI Evaluation ---
            node(
                func=evaluate_explanation_faithfulness,
                inputs=[
                    "cox_ph_model",
                    "survival_data_prepared",
                    "shap_explanation",
                    "parameters",
                ],
                outputs="faithfulness_score",
                name="evaluate_shap_faithfulness_node",
            ),
            node(
                func=evaluate_explanation_consistency,
                inputs=["shap_explanation", "survival_data_prepared"],
                outputs="consistency_score",
                name="evaluate_shap_consistency_node",
            ),
            # --- BTYD Models (AVOD/TVOD) ---
            node(
                func=train_bg_nbd_model,
                inputs=["processed_data", "parameters"],
                outputs="bg_nbd_model",
                name="train_bg_nbd_node",
            ),
            node(
                func=train_gamma_gamma_model,
                inputs=["processed_data", "parameters"],
                outputs="gamma_gamma_model",
                name="train_gamma_gamma_node",
            ),
            node(
                func=train_tweedie_model,
                inputs=["feature_store", "parameters"],
                outputs="tweedie_model",
                name="train_tweedie_node",
            ),
            # --- DL Sequential Model ---
            node(
                func=train_sequential_cltv_model,
                inputs=["tensor_sequences", "tensor_labels", "parameters"],
                outputs="sequential_cltv_model",
                name="train_sequential_cltv_node",
            ),
            # --- Inference & Business Metrics ---
            node(
                func=predict_cltv,
                inputs=[
                    "processed_data",
                    "feature_store",
                    "bg_nbd_model",
                    "gamma_gamma_model",
                    "cox_ph_model",
                    "sequential_cltv_model",
                    "tweedie_model",
                    "parameters",
                ],
                outputs="raw_cltv_predictions",
                name="predict_cltv_node",
            ),
            node(
                func=calculate_business_metrics,
                inputs=["raw_cltv_predictions", "feature_store"],
                outputs="churn_predictions",
                name="calculate_business_metrics_node",
            ),
        ]
    )
