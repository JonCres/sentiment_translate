import logging
import pandas as pd
from typing import Dict, Any
from transformers import pipeline
from utils.device import get_device, clear_device_cache

logger = logging.getLogger(__name__)

# Cache for the emotion pipeline
_emotion_pipeline = None


def load_emotion_model(model_name: str):
    """Load and cache the emotion analysis pipeline."""
    global _emotion_pipeline
    if _emotion_pipeline is not None:
        return _emotion_pipeline

    logger.info(f"Initializing Emotion Recognition with model: {model_name}")
    clear_device_cache()

    device_str = get_device(purpose="Emotion inference")
    if device_str == "cuda":
        device = 0
    elif device_str in ["xpu", "mps"]:
        device = device_str
    else:
        device = -1

    _emotion_pipeline = pipeline(
        "text-classification", model=model_name, return_all_scores=True, device=device
    )
    return _emotion_pipeline


def run_mer_inference(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Run Multimodal Emotion Recognition (MER / Emotion Analysis).

    Uses a pre-trained Transformer model to detect granular emotions
    and maps them to human-readable labels.

    Args:
        data: DataFrame with feedback text.
        parameters: MER model parameters.

    Returns:
        DataFrame with human-readable emotion labels and probabilities.
    """
    mer_params = parameters.get("mer", {})
    if not mer_params.get("enabled", True):
        logger.info("MER is disabled by configuration.")
        return data

    model_name = mer_params.get(
        "model_name", "j-hartmann/emotion-english-distilroberta-base"
    )
    label_map = mer_params.get("labels", {})

    # Load model
    emotion_pipe = load_emotion_model(model_name)

    logger.info("Running emotion analysis on feedback text...")

    texts = data["feedback_text_masked"].fillna("").astype(str).tolist()

    # Process in batches for efficiency
    batch_size = mer_params.get("batch_size", 16)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Transformers pipeline handles lists of strings
        batch_results = emotion_pipe(batch)
        results.extend(batch_results)

    # Convert results to a more usable format
    emotion_data = []
    for res in results:
        # res is a list of [{'label': 'joy', 'score': 0.9}, ...]
        scores = {f"emo_{item['label']}": item["score"] for item in res}

        # Determine dominant emotion and its human-readable label
        dominant_raw = max(res, key=lambda x: x["score"])["label"]
        scores["dominant_emotion_raw"] = dominant_raw
        scores["dominant_emotion"] = label_map.get(dominant_raw, dominant_raw.title())
        emotion_data.append(scores)

    emotion_df = pd.DataFrame(emotion_data, index=data.index)

    # Merge with original data
    return pd.concat([data, emotion_df], axis=1)
