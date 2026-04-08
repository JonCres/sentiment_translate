from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_bg_nbd_model, 
    train_gamma_gamma_model, 
    predict_cltv,
    train_sbg_model,
    train_weibull_aft_model,
    train_xgboost_residual_model,
    predict_contractual_cltv,
)
from .validation_nodes import run_kaplan_meier_analysis, run_cox_ph_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # --- Non-Contractual (BTYD) ---
            node(
                func=train_bg_nbd_model,
                inputs=["processed_data", "params:modeling.lifetimes.bg_nbd"],
                outputs="bg_nbd_model",
                name="train_bg_nbd_model_node",
            ),
            node(
                func=train_gamma_gamma_model,
                inputs=["processed_data", "params:modeling.lifetimes.gamma_gamma"],
                outputs="gamma_gamma_model",
                name="train_gamma_gamma_model_node",
            ),
            node(
                func=predict_cltv,
                inputs=[
                    "bg_nbd_model", 
                    "gamma_gamma_model", 
                    "processed_data",
                    "xgboost_model", # Optional
                    "feature_store"  # Optional
                ],
                outputs="cltv_predictions",
                name="predict_cltv_node",
            ),

            # --- Contractual (Survival) ---
            node(
                func=train_sbg_model,
                inputs=["survival_data", "params:modeling"],
                outputs="sbg_model",
                name="train_sbg_model_node",
            ),
            node(
                func=train_weibull_aft_model,
                inputs=["survival_data", "params:modeling"],
                outputs="weibull_model",
                name="train_weibull_aft_model_node",
            ),

            # --- Refinement (XGBoost) ---
            node(
                func=train_xgboost_residual_model,
                inputs=["feature_store", "parameters"],
                outputs="xgboost_model",
                name="train_xgboost_residual_model_node",
            ),

            # --- Contractual CLTV Prediction (SVOD Hybrid) ---
            node(
                func=predict_contractual_cltv,
                inputs=[
                    "sbg_model",
                    "weibull_model",
                    "xgboost_model",
                    "survival_data",
                    "feature_store"
                ],
                outputs="cltv_predictions_contractual",
                name="predict_contractual_cltv_node",
            ),
            
            # --- Validation ---
            node(
                func=run_kaplan_meier_analysis,
                inputs=["survival_data", "params:modeling"],
                outputs="kaplan_meier_result",
                name="run_kaplan_meier_analysis_node",
            ),
            node(
                func=run_cox_ph_analysis,
                inputs=["survival_data", "params:modeling"],
                outputs="cox_ph_result",
                name="run_cox_ph_analysis_node",
            ),
        ]
    )
