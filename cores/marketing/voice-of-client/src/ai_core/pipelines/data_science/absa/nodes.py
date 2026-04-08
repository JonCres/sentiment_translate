import logging
import pandas as pd
from typing import Dict, Any

from transformers import pipeline
from utils.device import get_device, clear_device_cache

logger = logging.getLogger(__name__)


def run_absa_inference(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Run Aspect-Based Sentiment Analysis on feedback text.

    Args:
        data: Features DataFrame with masked feedback text.
        parameters: Model hyperparameters from parameters.yml.

    Returns:
        DataFrame with sentiment scores per aspect.
    """
    if not parameters.get("absa", {}).get("enabled", True):
        logger.info("ABSA is disabled by configuration.")
        return data

    model_name = parameters["absa"]["model_name"]
    aspects = parameters["absa"]["aspects"]

    logger.info(f"Initializing Zero-Shot ABSA with model: {model_name}")
    clear_device_cache()

    device_str = get_device(purpose="ABSA inference")
    if device_str == "cuda":
        device = 0
    elif device_str in ["xpu", "mps"]:
        device = device_str
    else:
        device = -1

    # Zero-shot classification for aspect mapping
    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli", device=device
    )
    # Sentiment analysis for polarity (1-5 stars)
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
    )

    def get_aspect_sentiment(text: str) -> Dict[str, float]:
        if not text or len(str(text)) < 5:
            return {f"sent_{a}": 3.0 for a in aspects}

        # 1. Detect which aspects are mentioned
        results = classifier(text, candidate_labels=aspects, multi_label=True)
        relevant_aspects = {
            label: score
            for label, score in zip(results["labels"], results["scores"])
            if score > 0.3
        }

        # 2. Get overall sentiment as base (1-5 stars)
        sentiment_res = sentiment_analyzer(text[:512])[0]
        # label format is usually "1 star", "2 stars", etc.
        try:
            base_score = float(sentiment_res["label"].split()[0])
        except (ValueError, IndexError):
            base_score = 3.0

        aspect_scores = {}
        for aspect in aspects:
            if aspect in relevant_aspects:
                # If aspect is highly relevant, it inherits the sentiment polarity
                relevance = relevant_aspects[aspect]
                # Combine base sentiment with relevance
                score = base_score * 0.8 + (relevance * 2 + 3) * 0.2
                aspect_scores[f"sent_{aspect}"] = round(min(5.0, max(1.0, score)), 2)
            else:
                aspect_scores[f"sent_{aspect}"] = 3.0  # Neutral if not mentioned
        return aspect_scores

    sentiments = data["feedback_text_masked"].apply(get_aspect_sentiment)
    sentiment_df = pd.DataFrame(sentiments.tolist(), index=data.index)

    return pd.concat([data, sentiment_df], axis=1)
