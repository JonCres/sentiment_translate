from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_bg_nbd_model,
    train_gamma_gamma_model,
    predict_cltv,
    train_sbg_model,
    train_weibull_aft_model,
    train_xgboost_refinement,
    train_lstm_model,
    train_ensemble_model,
    predict_lstm_engagement,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
            # Optional: Contractual path (if data available)
            # node(
            #     func=train_sbg_model,
            #     inputs=["processed_data", "params:modeling.contractual.sbg"],
            #     outputs="sbg_model",
            #     name="train_sbg_model_node",
            # ),
            # node(
            #     func=predict_cltv,
            #     inputs=["bg_nbd_model", "gamma_gamma_model", "processed_data", "params:modeling"],
            #     outputs="cltv_predictions",
            #     name="predict_cltv_node",
            # ),
            node(
                func=train_lstm_model,
                inputs=["engagement_sequences", "params:modeling.lstm"],
                outputs="lstm_model",
                name="train_lstm_model_node",
            ),
            node(
                func=train_ensemble_model,
                inputs={
                    "bg_nbd_model": "bg_nbd_model",
                    "gamma_gamma_model": "gamma_gamma_model",
                    "lstm_model": "lstm_model",
                    "data": "processed_data",
                    "seq_data": "engagement_sequences",
                    "params": "params:modeling.ensemble"
                },
                outputs="ensemble_model",
                name="train_ensemble_model_node",
            ),
            node(
                func=predict_cltv,
                inputs={
                    "bg_nbd_model": "bg_nbd_model",
                    "gamma_gamma_model": "gamma_gamma_model",
                    "lstm_model": "lstm_model",
                    "ensemble_model": "ensemble_model",
                    "data": "processed_data",
                    "seq_data": "engagement_sequences",
                    "params": "params:modeling"
                },
                outputs="cltv_predictions",
                name="predict_cltv_node",
            ),
        ]
    )
