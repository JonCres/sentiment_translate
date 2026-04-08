from kedro.pipeline import Pipeline
from amazon_sentiment_project.pipelines  import speech_to_text, translation, validation, sentiment

# imports correctos (recomendado)
from amazon_sentiment_project.pipelines.translation.pipeline import create_pipeline as translation_pipeline
from amazon_sentiment_project.pipelines.validation.pipeline import create_pipeline as validation_pipeline
from amazon_sentiment_project.pipelines.sentiment.pipeline import create_pipeline as sentiment_pipeline
from amazon_sentiment_project.pipelines.speech_to_text.pipeline import create_pipeline as speech_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra todos los pipelines del proyecto
    """

    return {
        # PIPELINE PRINCIPAL (flujo completo)
        "__default__": (
            speech_pipeline()
            + translation_pipeline()
            + validation_pipeline()
            + sentiment_pipeline()
        ),

        # pipelines individuales
        "speech_to_text": speech_pipeline(),
        "translation": translation_pipeline(),
        "validation": validation_pipeline(),
        "sentiment": sentiment_pipeline(),
    }