from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["reviews_validated", "params:rating_prediction"],
                outputs=["X_train_rating", "X_test_rating", "y_train_rating", "y_test_rating"],
                name="split_rating_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train_rating", 
                    "X_test_rating", 
                    "y_train_rating", 
                    "y_test_rating",
                    "params:rating_prediction"
                ],
                outputs=["rating_prediction_model", "rating_prediction_metrics"],
                name="train_rating_model_node",
            ),
        ]
    )

def create_fine_tuning_pipeline(**kwargs) -> Pipeline:
    from .fine_tuning import fine_tune_embedding_model
    return pipeline(
        [
            node(
                func=fine_tune_embedding_model,
                inputs=["reviews_validated", "params:rating_prediction"],
                outputs="fine_tuned_embedding_model_path",
                name="fine_tune_embedding_model_node",
                tags=["fine_tuning"]
            )
        ]
    )
