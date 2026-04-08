"""Visualization pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_lifetimes_metrics,
    calculate_business_impact_kpis,
    plot_business_impact_kpis,
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
                func=calculate_business_impact_kpis,
                inputs=["cltv_predictions", "params:visualization"],
                outputs="business_kpis",
                name="calculate_business_impact_kpis_node",
            ),
            node(
                func=plot_business_impact_kpis,
                inputs=["business_kpis", "params:visualization"],
                outputs=None,
                name="plot_business_impact_kpis_node",
            ),
        ]
    )
