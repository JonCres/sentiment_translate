import polars as pl
import logging
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
from utils import get_device, clear_device_cache

logger = logging.getLogger(__name__)


def analyze_text_emotions(
    df: pl.DataFrame, model_params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Analyze emotions using a text-based transformer model.
    """
    logger.info("Analyzing text emotions...")
    clear_device_cache()

    model_name = model_params.get(
        "name", "j-hartmann/emotion-english-distilroberta-base"
    )
    batch_size = model_params.get("batch_size", 32)
    max_length = model_params.get("max_length", 512)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device_name = get_device("emotion analysis inference")
    device = torch.device(device_name)

    model.to(device)
    model.eval()

    # Text column from skeleton
    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )
    texts = df[text_col].to_list()

    all_emotions = []
    all_scores = []

    # Mapping indices to labels
    # Model: j-hartmann/emotion-english-distilroberta-base
    # Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    labels = model.config.id2label

    for i in tqdm(range(0, len(texts), batch_size), desc="Emotion Detection"):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.cpu().numpy()

        for pred in predictions:
            probs = softmax(pred)
            idx = probs.argmax()
            all_emotions.append(labels[idx])
            all_scores.append(float(probs[idx]))

    df_result = df.with_columns(
        [
            pl.Series("detected_emotion", all_emotions),
            pl.Series("emotion_confidence", all_scores),
        ]
    )

    logger.info(
        f"Emotion analysis complete. Distribution: {df_result['detected_emotion'].value_counts()}"
    )

    return df_result
