"""
ABSA (Aspect-Based Sentiment Analysis) Nodes

This module implements aspect-level sentiment analysis as described in
Section 1.1 of the Voice of Customer AI Technical Walkthrough.

Unlike document-level sentiment (which gives one score per review), ABSA:
1. Extracts aspect terms (e.g., "shipping", "quality", "price")
2. Classifies sentiment for each aspect separately
3. Allows understanding that a customer may love the "quality" but hate the "price"
"""

import polars as pl
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Try to import transformers for ABSA model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from utils import get_device, clear_device_cache

    ABSA_AVAILABLE = True
except ImportError:
    ABSA_AVAILABLE = False
    logger.warning("Transformers not available. ABSA functionality will be limited.")


# Predefined retail/CPG aspects commonly found in reviews
DEFAULT_ASPECTS = [
    "price",
    "quality",
    "shipping",
    "delivery",
    "packaging",
    "customer service",
    "value",
    "size",
    "color",
    "durability",
    "ease of use",
    "installation",
    "design",
    "performance",
    "material",
]


def extract_aspects(df: pl.DataFrame, aspect_params: Dict[str, Any]) -> pl.DataFrame:
    """
    Extract aspect terms from review text.

    Uses a combination of:
    1. Predefined aspect list (from parameters)
    2. Noun phrase extraction (if enabled)

    Args:
        df: DataFrame with review text
        aspect_params: Configuration for aspect extraction

    Returns:
        DataFrame with aspect_terms column (list of aspects per review)
    """
    logger.info("Extracting aspects from reviews...")

    # Determine text column (skeleton or legacy)
    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )

    # Get predefined aspects
    predefined = aspect_params.get("predefined", DEFAULT_ASPECTS)
    extract_custom = aspect_params.get("extract_custom", False)

    # Create aspect pattern for regex matching
    # Match whole words only (case insensitive)
    aspect_patterns = {
        aspect: re.compile(rf"\b{re.escape(aspect)}\b", re.IGNORECASE)
        for aspect in predefined
    }

    def find_aspects(text: str) -> List[str]:
        """Find all mentioned aspects in text."""
        if not text:
            return []

        found_aspects = []
        for aspect, pattern in aspect_patterns.items():
            if pattern.search(text):
                found_aspects.append(aspect)

        return found_aspects

    # Apply aspect extraction
    df = df.with_columns(
        pl.col(text_col)
        .map_elements(find_aspects, return_dtype=pl.List(pl.String))
        .alias("aspect_terms")
    )

    # Count reviews with aspects found
    reviews_with_aspects = df.filter(pl.col("aspect_terms").list.len() > 0)
    logger.info(
        f"Found aspects in {len(reviews_with_aspects)}/{len(df)} reviews "
        f"({len(reviews_with_aspects) / len(df) * 100:.1f}%)"
    )

    return df


def load_absa_model(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the ABSA sentiment model.

    Uses a DeBERTa-based model fine-tuned for aspect-based sentiment.

    Args:
        model_params: Model configuration (name, device, etc.)

    Returns:
        Dictionary containing tokenizer, model, and device
    """
    if not ABSA_AVAILABLE:
        logger.error("Transformers not available for ABSA")
        return None

    model_name = model_params.get("name", "yangheng/deberta-v3-base-absa-v1.1")
    logger.info(f"Loading ABSA model: {model_name}")
    clear_device_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Determine device
        device_name = get_device("ABSA model inference")
        device = torch.device(device_name)

        model.to(device)
        model.eval()

        logger.info(f"ABSA model loaded on device: {device}")

        return {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "model_name": model_name,
        }
    except Exception as e:
        logger.error(f"Failed to load ABSA model: {e}")
        return None


def analyze_aspect_sentiment(
    df: pl.DataFrame, absa_model: Optional[Dict[str, Any]], model_params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Analyze sentiment for each aspect in each review.

    For each review, creates aspect-sentiment pairs like:
    [{"aspect": "shipping", "sentiment": "positive", "score": 0.92},
     {"aspect": "price", "sentiment": "negative", "score": 0.78}]

    Args:
        df: DataFrame with aspect_terms column
        absa_model: Loaded ABSA model (tokenizer, model, device)
        model_params: Model parameters (batch_size, max_length)

    Returns:
        DataFrame with aspect_sentiments column
    """
    logger.info("Analyzing aspect-level sentiment...")

    text_col = (
        "Interaction_Payload" if "Interaction_Payload" in df.columns else "review_text"
    )

    if absa_model is None or not ABSA_AVAILABLE:
        # Fallback: Use simple lexicon-based approach
        logger.warning("ABSA model not available, using lexicon-based fallback")
        return _analyze_aspect_sentiment_lexicon(df, text_col)

    tokenizer = absa_model["tokenizer"]
    model = absa_model["model"]
    device = absa_model["device"]

    batch_size = model_params.get("batch_size", 16)
    max_length = model_params.get("max_length", 256)

    # Prepare data for aspect sentiment analysis
    all_aspect_sentiments = []

    # Get texts and aspects as lists
    texts = df[text_col].to_list()
    aspects_list = df["aspect_terms"].to_list()

    for i, (text, aspects) in enumerate(zip(texts, aspects_list)):
        if not aspects or not text:
            all_aspect_sentiments.append([])
            continue

        review_aspects = []

        # Process each aspect for this review
        for aspect in aspects:
            try:
                # ABSA format: "[CLS] text [SEP] aspect [SEP]"
                inputs = tokenizer(
                    text,
                    aspect,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)

                    # Get prediction (0=negative, 1=neutral, 2=positive)
                    pred_idx = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred_idx].item()

                    sentiment_labels = ["negative", "neutral", "positive"]
                    sentiment = sentiment_labels[pred_idx]

                    review_aspects.append(
                        {
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "confidence": round(confidence, 3),
                        }
                    )

            except Exception as e:
                logger.debug(f"Error processing aspect '{aspect}': {e}")
                # Add with unknown sentiment
                review_aspects.append(
                    {"aspect": aspect, "sentiment": "unknown", "confidence": 0.0}
                )

        all_aspect_sentiments.append(review_aspects)

        # Log progress
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(texts)} reviews for ABSA")

    # Add to dataframe
    df = df.with_columns(pl.Series("aspect_sentiments", all_aspect_sentiments))

    logger.info(f"ABSA complete: analyzed {len(df)} reviews")

    return df


def _analyze_aspect_sentiment_lexicon(df: pl.DataFrame, text_col: str) -> pl.DataFrame:
    """
    Fallback lexicon-based aspect sentiment analysis.

    Uses simple positive/negative word proximity to aspects.
    """
    # Simple sentiment lexicon
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "perfect",
        "love",
        "best",
        "awesome",
        "fantastic",
        "wonderful",
        "fast",
        "quick",
    }
    negative_words = {
        "bad",
        "poor",
        "terrible",
        "awful",
        "worst",
        "hate",
        "horrible",
        "slow",
        "broken",
        "disappointed",
        "wrong",
    }

    def analyze_aspects(row_data):
        text = row_data["text"]
        aspects = row_data["aspects"]

        if not aspects or not text:
            return []

        text_lower = text.lower()
        results = []

        for aspect in aspects:
            # Find text around the aspect
            aspect_idx = text_lower.find(aspect.lower())
            if aspect_idx == -1:
                continue

            # Get context window (50 chars before and after)
            start = max(0, aspect_idx - 50)
            end = min(len(text), aspect_idx + len(aspect) + 50)
            context = text_lower[start:end]

            # Count positive and negative words
            pos_count = sum(1 for w in positive_words if w in context)
            neg_count = sum(1 for w in negative_words if w in context)

            if pos_count > neg_count:
                sentiment = "positive"
                confidence = min(0.7 + 0.1 * pos_count, 0.95)
            elif neg_count > pos_count:
                sentiment = "negative"
                confidence = min(0.7 + 0.1 * neg_count, 0.95)
            else:
                sentiment = "neutral"
                confidence = 0.5

            results.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 3),
                }
            )

        return results

    # Convert to struct for processing
    struct_df = df.select(
        [pl.col(text_col).alias("text"), pl.col("aspect_terms").alias("aspects")]
    ).to_dicts()

    aspect_sentiments = [analyze_aspects(row) for row in struct_df]

    df = df.with_columns(pl.Series("aspect_sentiments", aspect_sentiments))

    return df


def create_aspect_summary(
    df: pl.DataFrame, summary_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create aggregate summary of aspect sentiment across all reviews.

    Args:
        df: DataFrame with aspect_sentiments column
        summary_params: Configuration for summary generation

    Returns:
        Dictionary with aspect-level insights
    """
    logger.info("Creating aspect sentiment summary...")

    # Flatten aspect sentiments
    aspect_data = []

    for row in df.select("aspect_sentiments").to_dicts():
        sentiments = row.get("aspect_sentiments", [])
        if sentiments:
            for item in sentiments:
                if isinstance(item, dict):
                    aspect_data.append(item)

    if not aspect_data:
        logger.warning("No aspect sentiments found")
        return {
            "total_aspects_analyzed": 0,
            "aspects": {},
            "timestamp": datetime.now().isoformat(),
        }

    # Aggregate by aspect
    aspect_summary = {}

    for item in aspect_data:
        aspect = item.get("aspect", "unknown")
        sentiment = item.get("sentiment", "unknown")
        confidence = item.get("confidence", 0.0)

        if aspect not in aspect_summary:
            aspect_summary[aspect] = {
                "total_mentions": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "unknown": 0,
                "avg_confidence": [],
            }

        aspect_summary[aspect]["total_mentions"] += 1
        aspect_summary[aspect][sentiment] += 1
        aspect_summary[aspect]["avg_confidence"].append(confidence)

    # Calculate percentages and average confidence
    for aspect, data in aspect_summary.items():
        total = data["total_mentions"]
        data["positive_pct"] = round(data["positive"] / total * 100, 1)
        data["neutral_pct"] = round(data["neutral"] / total * 100, 1)
        data["negative_pct"] = round(data["negative"] / total * 100, 1)
        data["avg_confidence"] = round(np.mean(data["avg_confidence"]), 3)

        # Net sentiment score: (positive - negative) / total
        data["net_sentiment"] = round((data["positive"] - data["negative"]) / total, 3)

    # Sort by total mentions
    sorted_aspects = dict(
        sorted(
            aspect_summary.items(), key=lambda x: x[1]["total_mentions"], reverse=True
        )
    )

    # Find problematic aspects (high negative %)
    problem_aspects = [
        aspect
        for aspect, data in sorted_aspects.items()
        if data["negative_pct"] > 30 and data["total_mentions"] >= 10
    ]

    # Find strong aspects (high positive %)
    strong_aspects = [
        aspect
        for aspect, data in sorted_aspects.items()
        if data["positive_pct"] > 70 and data["total_mentions"] >= 10
    ]

    summary = {
        "total_aspects_analyzed": len(aspect_data),
        "unique_aspects": len(sorted_aspects),
        "aspects": sorted_aspects,
        "problem_aspects": problem_aspects,
        "strong_aspects": strong_aspects,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"Aspect summary: {len(sorted_aspects)} unique aspects, "
        f"{len(problem_aspects)} problem areas, {len(strong_aspects)} strengths"
    )

    return summary
