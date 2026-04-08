"""Visualization pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_lifetimes_metrics, 
    plot_strategic_kpis, 
    interpret_cltv_visualizations, 
    save_cltv_interpretations,
    interpret_customer_prediction_cards,
    interpret_model_parameters_slm
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the visualization pipeline."""
    return pipeline(
        [
            node(
                func=plot_lifetimes_metrics,
                inputs=["bg_nbd_model", "processed_data", "params:visualization"],
                outputs=None,
                name="plot_lifetimes_metrics_node",
            ),
            node(
                func=plot_strategic_kpis,
                inputs=["cltv_predictions", "params:visualization"],
                outputs=None,
                name="plot_strategic_kpis_node",
            ),
            node(
                func=interpret_cltv_visualizations,
                inputs=["cltv_predictions", "params:visualization"],
                outputs="cltv_interpretations",
                name="interpret_cltv_visualizations_node",
            ),
            node(
                func=save_cltv_interpretations,
                inputs=["cltv_interpretations", "params:visualization"],
                outputs=None,
                name="save_cltv_interpretations_node",
            ),
            node(
                func=interpret_customer_prediction_cards,
                inputs=["cltv_predictions", "params:visualization"],
                outputs=None,
                name="interpret_customer_prediction_cards_node",
            ),
            node(
                func=interpret_model_parameters_slm,
                inputs=["bg_nbd_model", "gamma_gamma_model", "params:visualization"],
                outputs=None,
                name="interpret_model_parameters_slm_node",
            ),
        ]
    )
