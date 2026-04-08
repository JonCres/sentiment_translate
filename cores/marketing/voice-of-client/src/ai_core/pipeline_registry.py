# pipeline_registry.py
"""
Project pipelines.
"""

from typing import Dict
from kedro.pipeline import Pipeline

from .pipelines import data_processing as dp
from .pipelines import data_science as ds
from .pipelines import visualization as viz
from .pipelines import monitoring as mon


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    visualization_pipeline = viz.create_pipeline()
    monitoring_pipeline = mon.create_pipeline()

    return {
        "__default__": data_processing_pipeline
        + data_science_pipeline
        + visualization_pipeline
        + monitoring_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "churn": data_science_pipeline.only_nodes_with_tags("churn"),
        "visualization": visualization_pipeline,
        "monitoring": monitoring_pipeline,
    }
