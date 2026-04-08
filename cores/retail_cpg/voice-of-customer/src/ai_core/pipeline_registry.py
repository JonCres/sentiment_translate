from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from ai_core.pipelines import (
    data_processing,
    monitoring,
    visualization
)
from ai_core.pipelines.data_science import (
    sentiment_analysis,
    topic_modeling,
    rating_prediction,
    absa,
    mer,
    unified_insights
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    sentiment_analysis_pipeline = sentiment_analysis.create_pipeline()
    topic_modeling_pipeline = topic_modeling.create_pipeline()
    rating_prediction_pipeline = rating_prediction.create_pipeline()
    absa_pipeline = absa.create_pipeline()
    mer_pipeline = mer.create_pipeline()
    unified_insights_pipeline = unified_insights.create_pipeline()
    monitoring_pipeline = monitoring.create_pipeline()
    visualization_pipeline = visualization.create_pipeline()
    data_science_pipeline = (
        sentiment_analysis_pipeline 
        + topic_modeling_pipeline 
        + absa_pipeline 
        + mer_pipeline 
        + unified_insights_pipeline
    )

    return {
        "__default__": (
            data_processing_pipeline
            + data_science_pipeline
            + visualization_pipeline
            + monitoring_pipeline
        ),
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "sentiment_analysis": sentiment_analysis_pipeline,
        "topic_modeling": topic_modeling_pipeline,
        "rating_prediction": rating_prediction_pipeline,
        "absa": absa_pipeline,
        "mer": mer_pipeline,
        "unified_insights": unified_insights_pipeline,
        "visualization": visualization_pipeline,
        "monitoring": monitoring_pipeline,
    }
