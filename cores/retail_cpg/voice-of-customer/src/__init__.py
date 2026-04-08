from kedro.pipeline import Pipeline
from amazon_sentiment_project.pipelines.translation.pipeline import create_pipeline as translation_pipeline
from amazon_sentiment_project.pipelines.sentiment.pipeline import create_pipeline as sentiment_pipeline
from amazon_sentiment_project.pipelines.validation.pipeline import create_pipeline as validation_pipeline
from amazon_sentiment_project.pipelines.speech_to_text.pipeline import create_pipeline as transcribe_audio_pipeline

def register_pipelines():
    return {
        "__default__": translation_pipeline() + sentiment_pipeline() + validation_pipeline () + transcribe_audio_pipeline
    }