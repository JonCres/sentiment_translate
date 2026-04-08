from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_churn_predictor, predict_churn, explain_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_churn_predictor,
                inputs=["feedback_with_ratings", "params:modeling"],
                outputs=[
                    "churn_classifier",
                    "model_training_metrics",
                    "X_test_churn",
                    "y_test_churn",
                ],
                name="train_churn_predictor_node",
                tags=["churn"],
            ),
            node(
                func=predict_churn,
                inputs=["feedback_with_ratings", "churn_classifier"],
                outputs="feedback_with_churn",
                name="predict_churn_node",
                tags=["churn"],
            ),
            node(
                func=explain_predictions,
                inputs=["churn_classifier", "X_test_churn"],
                outputs="shap_explanations",
                name="explain_predictions_node",
                tags=["churn", "explainability"],
            ),
        ]
    )
