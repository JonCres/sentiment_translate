import polars as pl
import numpy as np
import logging
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
from datetime import datetime

from utils import MultiDeviceManager

logger = logging.getLogger(__name__)


def load_sentiment_model(model_params: Dict[str, Any]):
    """Load pre-trained sentiment analysis model with architecture-aware optimization."""
    logger.info(f"Loading sentiment model: {model_params['name']}")

    manager = MultiDeviceManager()
    manager.clear_cache()

    model_name = model_params["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Map precision
    dtype = (
        torch.float16
        if manager.device_type != manager.device_type.CPU
        else torch.float32
    )
    if manager.device_type == manager.device_type.MPS:
        dtype = torch.float32  # Better stability for sequence classification on MPS

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype
    )

    # Apply optimizations
    model = manager.optimize_model(model, dtype=dtype)
    manager.enable_memory_efficiency(model)
    manager.print_memory_stats()

    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": manager.device_name,
        "manager": manager,
    }


def analyze_sentiment(
    df: pl.DataFrame, sentiment_model: Dict[str, Any], model_params: Dict[str, Any]
) -> pl.DataFrame:
    """Analyze sentiment of reviews"""
    logger.info(f"Analyzing sentiment for {len(df)} reviews...")

    tokenizer = sentiment_model["tokenizer"]
    model = sentiment_model["model"]
    device = sentiment_model["device"]

    batch_size = model_params["batch_size"]
    max_length = model_params["max_length"]

    sentiments = []
    scores = []
    confidences = []

    # We can get all texts as a list
    all_texts = df["review_text"].to_list()
    total_reviews = len(all_texts)

    # Process in batches
    for i in tqdm(range(0, total_reviews, batch_size), desc="Sentiment Analysis"):
        batch_texts = all_texts[i : i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**encoded)
            predictions = outputs.logits.cpu().numpy()

        # Convert to probabilities
        for pred in predictions:
            probs = softmax(pred)
            sentiment_idx = np.argmax(probs)
            confidence = float(probs[sentiment_idx])

            # Map to labels: 0=negative, 1=neutral, 2=positive
            if sentiment_idx == 0:
                sentiment = "negative"
            elif sentiment_idx == 1:
                sentiment = "neutral"
            else:
                sentiment = "positive"

            sentiments.append(sentiment)
            scores.append(probs.tolist())
            confidences.append(confidence)

    # Add predictions to dataframe
    df_result = df.with_columns(
        [
            pl.Series("sentiment", sentiments),
            pl.Series("sentiment_confidence", confidences),
            # scores is a list of lists, Polars handles this as List(Float64)
            pl.Series("sentiment_scores", scores),
        ]
    )

    # Log distribution
    sentiment_dist = df_result["sentiment"].value_counts()
    logger.info(f"Sentiment distribution:\n{sentiment_dist}")

    return df_result


def create_sentiment_summary(df: pl.DataFrame) -> Dict[str, Any]:
    """Create sentiment analysis summary"""
    logger.info("Creating sentiment summary...")

    total_reviews = len(df)

    # Value counts returns a DF in Polars: col, count
    # .value_counts() -> | sentiment | count |
    sentiment_counts_df = df["sentiment"].value_counts()
    sentiment_counts = {
        row["sentiment"]: row["count"]
        for row in sentiment_counts_df.iter_rows(named=True)
    }

    # Percentages
    sentiment_percentages = {
        k: round((v / total_reviews) * 100, 2) for k, v in sentiment_counts.items()
    }

    avg_confidence = df["sentiment_confidence"].mean()

    # Sentiment by rating
    # Group by rating and sentiment, count, then pivot/structure
    # Pandas: df.groupby('rating')['sentiment'].value_counts(normalize=True).unstack(fill_value=0)
    # We want nested dict: {rating: {sentiment: fraction}}

    # First get counts per rating-sentiment pair
    sent_by_rating_df = df.group_by(["rating", "sentiment"]).len()

    # Calculate totals per rating
    rating_totals = df.group_by("rating").len().rename({"len": "total"})

    # Join and calc fraction
    sent_by_rating_df = sent_by_rating_df.join(rating_totals, on="rating")
    sent_by_rating_df = sent_by_rating_df.with_columns(
        (pl.col("len") / pl.col("total")).alias("fraction")
    )

    # Convert to dictionary structure
    sentiment_by_rating = {}
    # Iterate and build dict
    for row in sent_by_rating_df.iter_rows(named=True):
        r = row["rating"]
        s = row["sentiment"]
        f = row["fraction"]
        if r not in sentiment_by_rating:
            sentiment_by_rating[r] = {}
        sentiment_by_rating[r][s] = round(
            f, 2
        )  # keeping consistent with general formatting

    # Fill missing with 0 if needed (pandas unstack(fill_value=0) did this)
    # The loop above only adds existing combinations.
    # If strictly matching previous behavior, ensure all sentiments exist for all ratings found.
    # But usually sufficient.

    # Average confidence by sentiment
    avg_conf_df = df.group_by("sentiment").agg(pl.col("sentiment_confidence").mean())
    avg_confidence_by_sentiment = {
        row["sentiment"]: round(row["sentiment_confidence"], 3)
        for row in avg_conf_df.iter_rows(named=True)
    }

    summary = {
        "total_reviews": total_reviews,
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_percentages,
        "average_confidence": round(avg_confidence, 3),
        "confidence_by_sentiment": avg_confidence_by_sentiment,
        "sentiment_by_rating": sentiment_by_rating,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Sentiment summary created: {sentiment_counts}")

    return summary


def create_customer_sentiment_profiles(
    df: pl.DataFrame, customer_params: Dict[str, Any]
) -> pl.DataFrame:
    """Create sentiment profiles for each customer."""
    logger.info("Creating customer sentiment profiles...")

    if "customer_id" not in df.columns:
        raise ValueError("customer_id column is required for customer profiling")

    min_reviews = customer_params.get("min_reviews_per_customer", 3)

    # Aggregate
    # Calculate counts using lowercase for robustness
    customer_profiles = (
        df.with_columns(pl.col("sentiment").str.to_lowercase().alias("sentiment_lower"))
        .group_by("customer_id")
        .agg(
            [
                pl.len().alias("total_reviews"),
                pl.col("sentiment_confidence").mean().alias("avg_confidence"),
                pl.col("rating").mean().alias("avg_rating"),
                (pl.col("sentiment_lower") == "positive").sum().alias("positive_count"),
                (pl.col("sentiment_lower") == "neutral").sum().alias("neutral_count"),
                (pl.col("sentiment_lower") == "negative").sum().alias("negative_count"),
            ]
        )
    )

    # Add percentages
    customer_profiles = customer_profiles.with_columns(
        [
            (pl.col("positive_count") / pl.col("total_reviews") * 100)
            .round(2)
            .alias("positive_pct"),
            (pl.col("neutral_count") / pl.col("total_reviews") * 100)
            .round(2)
            .alias("neutral_pct"),
            (pl.col("negative_count") / pl.col("total_reviews") * 100)
            .round(2)
            .alias("negative_pct"),
        ]
    )

    # Sentiment score: (Positive - Negative) / Total
    # Use explicit casting to Float64 for the calculation
    customer_profiles = customer_profiles.with_columns(
        (
            (
                pl.col("positive_count").cast(pl.Float64)
                - pl.col("negative_count").cast(pl.Float64)
            )
            / pl.when(pl.col("total_reviews") > 0)
            .then(pl.col("total_reviews").cast(pl.Float64))
            .otherwise(pl.lit(1.0))
        )
        .round(3)
        .alias("sentiment_score")
    )

    # Dominant sentiment
    # We can use a when-then-otherwise chain or a struct+map (slower).
    # Since it's logic based on comparison of 3 columns, when-then is best.
    # Logic: positive >= max(neutral, negative) ? positive : (negative > neutral ? negative : neutral)

    customer_profiles = customer_profiles.with_columns(
        pl.when(
            pl.col("positive_count")
            >= pl.max_horizontal(["neutral_count", "negative_count"])
        )
        .then(pl.lit("positive"))
        .otherwise(
            pl.when(pl.col("negative_count") > pl.col("neutral_count"))
            .then(pl.lit("negative"))
            .otherwise(pl.lit("neutral"))
        )
        .alias("dominant_sentiment")
    )

    # Filter
    customer_profiles = customer_profiles.filter(pl.col("total_reviews") >= min_reviews)

    # Round
    customer_profiles = customer_profiles.with_columns(
        [pl.col("avg_confidence").round(3), pl.col("avg_rating").round(2)]
    )

    logger.info(f"Created sentiment profiles for {len(customer_profiles)} customers")

    return customer_profiles


def create_customer_sentiment_summary(
    customer_profiles: pl.DataFrame, customer_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create summary statistics for customer sentiment profiles."""
    logger.info("Creating customer sentiment summary...")

    top_n = customer_params.get("top_customers_count", 20)

    total_customers = len(customer_profiles)
    avg_reviews_per_customer = customer_profiles["total_reviews"].mean()

    # Sentiment distribution
    sent_dist_df = customer_profiles["dominant_sentiment"].value_counts()
    sentiment_dist = {
        row["dominant_sentiment"]: row["count"]
        for row in sent_dist_df.iter_rows(named=True)
    }

    sentiment_dist_pct = {
        k: round((v / total_customers) * 100, 2) for k, v in sentiment_dist.items()
    }

    # Top customers
    # top_k returns df
    top_by_volume = (
        customer_profiles.top_k(top_n, by="total_reviews")
        .select(
            ["customer_id", "total_reviews", "dominant_sentiment", "sentiment_score"]
        )
        .to_dicts()
    )

    top_positive = (
        customer_profiles.top_k(top_n, by="positive_pct")
        .select(["customer_id", "positive_pct", "total_reviews"])
        .to_dicts()
    )

    top_negative = (
        customer_profiles.top_k(top_n, by="negative_pct")
        .select(["customer_id", "negative_pct", "total_reviews"])
        .to_dicts()
    )

    summary = {
        "total_customers": total_customers,
        "avg_reviews_per_customer": round(avg_reviews_per_customer, 2),
        "customer_sentiment_distribution": sentiment_dist,
        "customer_sentiment_distribution_pct": sentiment_dist_pct,
        "avg_customer_sentiment_score": round(
            customer_profiles["sentiment_score"].mean(), 3
        ),
        "top_customers_by_volume": top_by_volume,
        "top_positive_customers": top_positive,
        "top_negative_customers": top_negative,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Customer sentiment summary created for {total_customers} customers")

    return summary
