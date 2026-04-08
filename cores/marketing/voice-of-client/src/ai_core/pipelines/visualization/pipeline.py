"""Visualization pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_feature_distribution_plots,
    create_correlation_heatmap,
    create_model_evaluation_plots,
    create_analysis_visualizations,
    generate_feedback_level_insights,
    generate_temporal_insights,
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
                inputs=[
                    "final_enriched_feedback",
                    "params:visualization.feature_dist",
                    "params:visualization.slm",
                ],
                outputs="feature_dist_plot",
                name="create_feature_distributions_node",
            ),
            node(
                func=create_correlation_heatmap,
                inputs=[
                    "final_enriched_feedback",
                    "params:visualization.correlation",
                    "params:visualization.slm",
                ],
                outputs="correlation_plot",
                name="create_correlation_heatmap_node",
            ),
            node(
                func=create_model_evaluation_plots,
                inputs=[
                    "churn_classifier",
                    "X_test_churn",
                    "y_test_churn",
                    "params:visualization.model_eval",
                    "params:visualization.slm",
                ],
                outputs="model_eval_plot",
                name="create_model_evaluation_node",
            ),
            node(
                func=create_analysis_visualizations,
                inputs=[
                    "nlp_business_correlations",
                    "params:visualization.analysis",
                    "params:visualization.slm",
                ],
                outputs="analysis_plots",
                name="create_analysis_visualizations_node",
            ),
            node(
                func=generate_feedback_level_insights,
                inputs=["final_enriched_feedback", "params:visualization.analysis"],
                outputs="feedback_insights",
                name="generate_feedback_insights_node",
            ),
            node(
                func=generate_temporal_insights,
                inputs=["final_enriched_feedback", "params:visualization.analysis"],
                outputs="temporal_insights",
                name="generate_temporal_insights_node",
            ),
            node(
                func=save_visualizations,
                inputs=[
                    "feature_dist_plot",
                    "correlation_plot",
                    "model_eval_plot",
                    "analysis_plots",
                    "feedback_insights",
                    "temporal_insights",
                    "params:visualization.save",
                ],
                outputs=None,
                name="save_visualizations_node",
            ),
        ]
    )
