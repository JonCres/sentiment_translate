"""Visualization pipeline definition."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_feature_distribution_plots,
    create_correlation_heatmap,
    create_model_evaluation_plots,
    save_visualizations,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the visualization pipeline.
    
    Returns:
        A Pipeline object containing visualization nodes.
    """
    return pipeline(
        [
            node(
                func=create_feature_distribution_plots,
                inputs=["features_data", "params:visualization.feature_dist"],
                outputs="feature_dist_plot",
                name="create_feature_distributions_node",
            ),
            node(
                func=create_correlation_heatmap,
                inputs=["features_data", "params:visualization.correlation"],
                outputs="correlation_plot",
                name="create_correlation_heatmap_node",
            ),
            node(
                func=create_model_evaluation_plots,
                inputs=["regressor", "X_test", "y_test", "params:visualization.model_eval"],
                outputs="model_eval_plot",
                name="create_model_evaluation_node",
            ),
            node(
                func=save_visualizations,
                inputs=[
                    "feature_dist_plot",
                    "correlation_plot",
                    "model_eval_plot",
                    "params:visualization.save",
                ],
                outputs=None,
                name="save_visualizations_node",
            ),
        ]
    )
