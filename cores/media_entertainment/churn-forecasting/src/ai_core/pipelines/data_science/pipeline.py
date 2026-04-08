from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    predict_churn,
    calculate_business_metrics,
    train_catboost_model,
    train_ensemble_model,
    train_lightgbm_model,
    train_random_forest_model,
    train_xgboost_residual_model,
    explain_model_shap,
    explain_model_lime,
    prepare_training_data,
    split_data,
)
from .dl_nodes import train_deep_learning_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict_churn,
                inputs=[
                    "ensemble_model",  # Now primary model
                    "feature_store",
                    "tensor_sequences",
                    "churn_lstm_model",
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
            # --- Deep Learning ---
            node(
                func=train_deep_learning_model,
                inputs=[
                    "tensor_sequences",
                    "tensor_labels",
                    "params:modeling",
                ],
                outputs="churn_lstm_model",
                name="train_deep_learning_model_node",
            ),
            # --- Data Prep & Split ---
            node(
                func=prepare_training_data,
                inputs=["feature_store", "parameters"],
                outputs="prepared_data_pd",
                name="prepare_training_data_node",
            ),
            node(
                func=split_data,
                inputs=["prepared_data_pd", "parameters"],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
            # --- Refinement (Ensemble) ---
            node(
                func=train_xgboost_residual_model,
                inputs=["train_data", "test_data", "parameters"],
                outputs="xgboost_model",
                name="train_xgboost_node",
            ),
            node(
                func=train_random_forest_model,
                inputs=["train_data", "test_data", "parameters"],
                outputs="random_forest_model",
                name="train_rf_node",
            ),
            node(
                func=train_lightgbm_model,
                inputs=["train_data", "test_data", "parameters"],
                outputs="lightgbm_model",
                name="train_lgb_node",
            ),
            node(
                func=train_catboost_model,
                inputs=["train_data", "test_data", "parameters"],
                outputs="catboost_model",
                name="train_cb_node",
            ),
            node(
                func=train_ensemble_model,
                inputs=[
                    "xgboost_model",
                    "random_forest_model",
                    "lightgbm_model",
                    "catboost_model",
                    "parameters",
                ],
                outputs="ensemble_model",
                name="train_ensemble_node",
            ),
            # --- Explainability ---
            node(
                func=explain_model_shap,
                inputs=["ensemble_model", "train_data", "parameters"],
                outputs="shap_explanation",
                name="explain_shap_node",
            ),
            node(
                func=explain_model_lime,
                inputs=["ensemble_model", "train_data", "parameters"],
                outputs="lime_explanation",
                name="explain_lime_node",
            ),
        ]
    )
