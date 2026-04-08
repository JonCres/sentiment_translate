import logging
import pandas as pd
from typing import Dict, Any
from transformers import pipeline
from utils import MultiDeviceManager

logger = logging.getLogger(__name__)

# Global cache for the sentiment model
_sentiment_scoring_pipeline = None


def load_context_aware_model(model_name: str):
    """Load and cache the context-aware sentiment scoring model using MultiDeviceManager."""
    global _sentiment_scoring_pipeline
    if _sentiment_scoring_pipeline is not None:
        return _sentiment_scoring_pipeline

    logger.info(
        f"Initializing Context-Aware Sentiment Scoring with model: {model_name}"
    )

    manager = MultiDeviceManager()
    manager.clear_cache()

    # Determine device for transformers pipeline
    # Transformers pipeline 'device' arg:
    # - integer for cuda device ID
    # - string for 'mps' or 'xpu'
    # - -1 for cpu
    if manager.device_type == manager.device_type.CUDA:
        device = 0
    elif manager.device_type in [manager.device_type.XPU, manager.device_type.MPS]:
        device = manager.device_name
    else:
        device = -1

    # Using a model that provides polarity and intensity
    _sentiment_scoring_pipeline = pipeline(
        "sentiment-analysis", model=model_name, return_all_scores=True, device=device
    )

    logger.info(f"Sentiment scoring pipeline initialized on {manager.device_name}")
    return _sentiment_scoring_pipeline


def run_context_aware_scoring(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Run Model 4: Context-Aware Sentiment Scoring.

    Assigns nuanced sentiment polarity and intensity scores based on linguistic context.
    Handles professional tone and technical terminology common in B2B.
    """
    scoring_params = parameters.get("sentiment_scoring", {})
    if not scoring_params.get("enabled", True):
        logger.info("Context-Aware Sentiment Scoring is disabled.")
        return data

    model_name = scoring_params.get(
        "model_name", "ProsusAI/finbert"
    )  # Good for professional/B2B context

    # Load model
    sentiment_pipe = load_context_aware_model(model_name)

    logger.info("Generating context-aware sentiment scores...")
    texts = data["feedback_text_masked"].fillna("").astype(str).tolist()

    # Process in batches
    batch_size = scoring_params.get("batch_size", 16)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if not batch:
            continue
        # Truncate to model max length
        batch = [t[:512] for t in batch]
        batch_results = sentiment_pipe(batch)
        results.extend(batch_results)

    # Convert to continuous score: -1 (Negative) to 1 (Positive)
    # FinBERT labels: neutral, positive, negative
    sentiment_scores = []
    for res in results:
        # res is a list of [{'label': 'positive', 'score': 0.9}, ...]
        scores_dict = {item["label"].lower(): item["score"] for item in res}

        # Calculate continuous polarity
        # Formula: pos_score - neg_score
        pos = scores_dict.get("positive", 0)
        neg = scores_dict.get("negative", 0)
        neu = scores_dict.get("neutral", 0)

        # Weighted polarity
        polarity = pos - neg

        # Intensity: how sure the model is (confidence)
        intensity = 1.0 - neu

        sentiment_scores.append(
            {
                "sentiment_polarity": round(float(polarity), 4),
                "sentiment_intensity": round(float(intensity), 4),
                "context_score": round(
                    float(polarity * intensity), 4
                ),  # Combined metric
            }
        )

    scoring_df = pd.DataFrame(sentiment_scores, index=data.index)

    # Concatenate with original data
    return pd.concat([data, scoring_df], axis=1)
