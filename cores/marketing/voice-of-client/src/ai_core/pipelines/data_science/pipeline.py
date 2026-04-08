from kedro.pipeline import Pipeline, pipeline
from .absa.pipeline import create_pipeline as create_absa_pipeline
from .mer.pipeline import create_pipeline as create_mer_pipeline
from .sentiment_scoring.pipeline import create_pipeline as create_scoring_pipeline
from .topic_modeling.pipeline import create_pipeline as create_topic_pipeline
from .churn_prediction.pipeline import create_pipeline as create_churn_pipeline
from .ratings_prediction.pipeline import create_pipeline as create_ratings_pipeline


from .analysis.pipeline import create_pipeline as create_analysis_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create the unified Data Science pipeline with modular sub-pipelines.

    Each component can be toggled via parameters.
    """
    absa_pipe = create_absa_pipeline()
    mer_pipe = create_mer_pipeline()
    scoring_pipe = create_scoring_pipeline()
    ratings_pipe = create_ratings_pipeline()
    topic_pipe = create_topic_pipeline()
    churn_pipe = create_churn_pipeline()
    analysis_pipe = create_analysis_pipeline()

    # The pipeline flow is: ABSA -> MER -> Scoring -> Ratings -> Churn -> Topics -> Analysis
    # We combine them into a single pipeline.
    # We rewire ratings_pipe to take the output of scoring_pipe
    rewired_ratings = pipeline(
        ratings_pipe,
        inputs={"feedback_with_emotions": "feedback_with_context_sentiment"},
    )

    ds_pipeline = (
        absa_pipe 
        + mer_pipe 
        + scoring_pipe 
        + rewired_ratings 
        + churn_pipe 
        + topic_pipe 
        + analysis_pipe
    )

    return ds_pipeline
