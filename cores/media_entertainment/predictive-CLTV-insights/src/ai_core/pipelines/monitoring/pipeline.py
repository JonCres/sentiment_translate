from kedro.pipeline import Pipeline, node, pipeline
from .nodes import monitor_data_drift, monitor_model_performance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=monitor_data_drift,
                inputs=["processed_data", "params:mlops.drift_detection"],
                outputs="drift_metrics",
                name="monitor_data_drift_node",
            ),
            node(
                func=monitor_model_performance,
                inputs=["cltv_predictions", "params:mlops"],
                outputs="performance_metrics",
                name="monitor_model_performance_node",
            ),
        ]
    )
