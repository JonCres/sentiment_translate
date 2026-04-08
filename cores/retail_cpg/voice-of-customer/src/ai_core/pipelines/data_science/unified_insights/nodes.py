import polars as pl
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

def create_unified_interaction_record(
    sentiment_df: pl.DataFrame,
    aspects_df: pl.DataFrame,
    topics_df: pl.DataFrame,
    emotions_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Combine all model outputs into a unified interaction record matching
    the Sample Output Record format.
    """
    logger.info("Creating unified interaction records...")
    
    # Join all DataFrames on Interaction_ID (skeleton) or review_id (original)
    # We'll use review_id for now as it's the common key in existing data science outputs
    # If using skeleton, it would be Interaction_ID
    
    id_col = 'review_id' if 'review_id' in sentiment_df.columns else 'Interaction_ID'
    
    unified_df = (
        sentiment_df.select([id_col, 'customer_id', 'product_id', 'date', 'sentiment', 'sentiment_confidence', 'review_text'])
        .join(aspects_df.select([id_col, 'aspect_sentiments']), on=id_col, how='left')
        .join(topics_df.select([id_col, 'topic_name']), on=id_col, how='left')
        .join(emotions_df.select([id_col, 'detected_emotion', 'emotion_confidence']), on=id_col, how='left')
    )
    
    # Fill nulls for safety
    unified_df = unified_df.with_columns([
        pl.col('aspect_sentiments').fill_null([]),
        pl.col('topic_name').fill_null("Uncategorized"),
        pl.col('detected_emotion').fill_null("neutral"),
        pl.col('emotion_confidence').fill_null(0.0)
    ])
    
    # Derive Actionable Insights (Logic from plan)
    # 1. Urgency
    unified_df = unified_df.with_columns(
        pl.when(
            (pl.col('sentiment') == 'negative') & 
            (pl.col('sentiment_confidence') > 0.8) | 
            (pl.col('detected_emotion').is_in(['anger', 'fear']))
        ).then(pl.lit('High'))
        .otherwise(
            pl.when(pl.col('sentiment') == 'negative').then(pl.lit('Medium'))
            .otherwise(pl.lit('Low'))
        ).alias('urgency')
    )
    
    # 2. Recommended Action
    # Simple rule-based mapping
    action_map = {
        ("Shipping", "negative"): "Check Logistics / Delivery Status",
        ("Price", "negative"): "Review Pricing Strategy / Offer Discount",
        ("Quality", "negative"): "Alert Product Quality Team",
        ("Customer Service", "negative"): "Escalate to Senior Support Agent",
    }
    
    def derive_action(topic: str, sentiment: str, emotion: str) -> str:
        if sentiment == 'positive':
            return "Send Thank You Note / Promote Loyalty"
        
        # Check specific mappings
        for (t, s), action in action_map.items():
            if t.lower() in topic.lower() and s == sentiment:
                return action
        
        if emotion == 'anger':
            return "Immediate Crisis Management Outreach"
            
        return "Monitor and Categorize"

    unified_df = unified_df.with_columns(
        pl.struct(['topic_name', 'sentiment', 'detected_emotion']).map_elements(
            lambda x: derive_action(x['topic_name'], x['sentiment'], x['detected_emotion']),
            return_dtype=pl.String
        ).alias('recommended_action')
    )
    
    # Format for Sample Output Record structure (as a JSON-like struct)
    # This makes it easier to export or use in UI
    unified_df = unified_df.with_columns(
        pl.struct([
            pl.col(id_col).alias('interaction_id'),
            pl.col('customer_id'),
            pl.col('date').alias('timestamp'),
            pl.col('review_text').alias('payload'),
            pl.col('sentiment').alias('sentiment_label'),
            pl.col('sentiment_confidence').alias('confidence'),
            pl.col('detected_emotion').alias('emotions_detected'),
            pl.col('aspect_sentiments'),
            pl.col('topic_name').alias('topics'),
            pl.col('recommended_action'),
            pl.col('urgency')
        ]).alias('sample_output_record')
    )
    
    logger.info(f"Created {len(unified_df)} unified records")
    return unified_df

def create_customer_aspect_profiles(
    aspects_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Aggregates aspect sentiment at the customer level.
    """
    logger.info("Creating customer aspect profiles...")
    
    # Flatten aspect_sentiments
    # aspects_df has col 'aspect_sentiments' which is List(Struct)
    # Struct contains: aspect, sentiment, confidence
    
    flattened = aspects_df.select(['customer_id', 'aspect_sentiments']).explode('aspect_sentiments')
    
    # Filter out empty/null aspects
    flattened = flattened.filter(pl.col('aspect_sentiments').is_not_null())
    
    # Extract struct fields
    flattened = flattened.with_columns([
        pl.col('aspect_sentiments').struct.field('aspect').alias('aspect'),
        pl.col('aspect_sentiments').struct.field('sentiment').alias('sentiment'),
        pl.col('aspect_sentiments').struct.field('confidence').alias('confidence')
    ])
    
    # Convert sentiment to score: positive=1, neutral=0, negative=-1
    flattened = flattened.with_columns(
        pl.col('sentiment').replace({
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }, default=0.0).alias('sentiment_score')
    )
    
    # Aggregate by customer and aspect
    profiles = flattened.group_by(['customer_id', 'aspect']).agg([
        pl.len().alias('mention_count'),
        pl.col('sentiment_score').mean().alias('avg_sentiment_score'),
        pl.col('confidence').mean().alias('avg_confidence')
    ])
    
    logger.info(f"Created aspect profiles for {profiles['customer_id'].n_unique()} customers")
    return profiles
