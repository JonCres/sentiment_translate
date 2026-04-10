"""
End-to-End tests for the Voice of Customer data pipeline.

These tests exercise the full data processing chain from raw input
through skeleton mapping, validation, cleaning, feature engineering,
aspect extraction, monitoring, and unified insights generation.
"""

import sys
from unittest.mock import MagicMock

# Stub heavy / network-bound transitive dependencies BEFORE any project import
sys.modules.setdefault("deep_translator", MagicMock())
sys.modules.setdefault("langdetect", MagicMock())
sys.modules.setdefault("utils", MagicMock())
sys.modules.setdefault("mlflow", MagicMock())

import polars as pl
import pytest
from datetime import datetime

from ai_core.pipelines.data_processing.nodes import (
    map_to_skeleton,
    validate_reviews,
    clean_reviews,
    engineer_features,
    translate_reviews,
)
from ai_core.pipelines.data_science.absa.nodes import (
    extract_aspects,
    _analyze_aspect_sentiment_lexicon,
    create_aspect_summary,
)
from ai_core.pipelines.data_science.unified_insights.nodes import (
    create_unified_interaction_record,
)
from ai_core.pipelines.monitoring.nodes import (
    check_sentiment_drift,
    check_topic_drift,
    generate_alerts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_amazon_reviews():
    """Simulates raw Amazon review data before skeleton mapping."""
    return pl.DataFrame({
        "review_id": [f"r{i}" for i in range(1, 11)],
        "product_id": [f"p{i % 3 + 1}" for i in range(1, 11)],
        "customer_id": [f"c{i % 4 + 1}" for i in range(1, 11)],
        "review_text": [
            "Great quality product! Fast shipping and good packaging.",
            "Terrible experience. Poor quality and slow delivery.",
            "It's okay. Average price for what you get.",
            "Love the design! Amazing performance and durability.",
            "Disappointed with shipping. Broke after a week.",
            "<b>Excellent</b> customer service! Worth every penny.",
            "Not great. Bad size and wrong color. http://example.com",
            "Perfect quality!!!  Extra   spaces   here.",
            "Good value for the price. Easy installation.",
            "Worst purchase ever. Horrible packaging.",
        ],
        "rating": [5.0, 1.0, 3.0, 5.0, 1.0, 5.0, 2.0, 4.0, 4.0, 1.0],
        "date": [datetime(2024, 1, i + 1) for i in range(10)],
    })


@pytest.fixture
def mapping_params():
    return {
        "mandatory": {
            "Interaction_ID": "review_id",
            "Interaction_Payload": "review_text",
            "Customer_ID": "customer_id",
            "Timestamp": "date",
            "Target_Object_ID": "product_id",
            "Rating": "rating",
        },
        "optional": {},
        "defaults": {
            "Channel_ID": "Web_Review",
            "Language_Code": "en",
        },
    }


@pytest.fixture
def validation_params():
    return {
        "required_columns": [
            "Interaction_ID", "Interaction_Payload", "Rating",
        ],
        "min_review_length": 5,
        "max_review_length": 5000,
        "rating_range": [1.0, 5.0],
        "max_null_percentage": 0.5,
    }


@pytest.fixture
def cleaning_params():
    return {
        "remove_html": True,
        "remove_urls": True,
        "lowercase": True,
        "remove_extra_spaces": True,
    }


@pytest.fixture
def feature_params():
    return {
        "extract_length": True,
        "extract_word_count": True,
        "extract_exclamation_count": True,
        "extract_question_count": True,
        "extract_caps_ratio": True,
    }


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestFullDataProcessingPipeline:
    """E2E: raw → skeleton → validate → clean → features."""

    def test_full_pipeline_produces_features(
        self, raw_amazon_reviews, mapping_params, validation_params,
        cleaning_params, feature_params,
    ):
        skeleton = map_to_skeleton(raw_amazon_reviews, mapping_params)
        validated = validate_reviews(skeleton, validation_params)
        cleaned = clean_reviews(validated, cleaning_params)
        features = engineer_features(cleaned, feature_params)

        # Skeleton columns still present
        for col in ("Interaction_ID", "Interaction_Payload", "Customer_ID",
                     "Rating", "Channel_ID", "Language_Code"):
            assert col in features.columns

        # Engineered feature columns present
        for col in ("review_length", "word_count", "exclamation_count",
                     "question_count", "caps_ratio"):
            assert col in features.columns

        assert len(features) > 0

    def test_pipeline_preserves_row_integrity(
        self, raw_amazon_reviews, mapping_params, validation_params,
        cleaning_params, feature_params,
    ):
        skeleton = map_to_skeleton(raw_amazon_reviews, mapping_params)
        validated = validate_reviews(skeleton, validation_params)
        cleaned = clean_reviews(validated, cleaning_params)
        features = engineer_features(cleaned, feature_params)

        # No rows should have null Interaction_ID
        assert features["Interaction_ID"].null_count() == 0


@pytest.mark.e2e
class TestTranslationPipeline:
    """E2E: skeleton → translate (disabled) → validate → clean → features."""

    def test_translation_disabled_passthrough(
        self, raw_amazon_reviews, mapping_params, validation_params,
        cleaning_params, feature_params,
    ):
        skeleton = map_to_skeleton(raw_amazon_reviews, mapping_params)

        translated = translate_reviews(skeleton, {
            "enabled": False,
            "target_language": "en",
        })

        validated = validate_reviews(translated, validation_params)
        cleaned = clean_reviews(validated, cleaning_params)
        features = engineer_features(cleaned, feature_params)

        assert len(features) > 0
        assert "review_length" in features.columns


@pytest.mark.e2e
class TestABSAPipeline:
    """E2E: reviews → extract_aspects → lexicon sentiment → summary."""

    def test_full_absa_pipeline(self, raw_amazon_reviews, mapping_params):
        skeleton = map_to_skeleton(raw_amazon_reviews, mapping_params)

        aspect_params = {"predefined": ["quality", "shipping", "price",
                                         "design", "packaging", "delivery"],
                         "extract_custom": False}
        with_aspects = extract_aspects(skeleton, aspect_params)

        assert "aspect_terms" in with_aspects.columns

        with_sentiments = _analyze_aspect_sentiment_lexicon(
            with_aspects, "Interaction_Payload"
        )
        assert "aspect_sentiments" in with_sentiments.columns

        summary = create_aspect_summary(with_sentiments, {})

        assert "total_aspects_analyzed" in summary
        assert "unique_aspects" in summary
        assert summary["total_aspects_analyzed"] >= 0


@pytest.mark.e2e
class TestMonitoringPipeline:
    """E2E: sentiment summary → drift checks → alert generation."""

    def test_monitoring_with_drift(self):
        summary = {
            "sentiment_percentages": {
                "positive": 80.0,
                "neutral": 10.0,
                "negative": 10.0,
            },
            "topic_distribution": {"quality": 50, "shipping": 30},
        }
        params = {
            "sentiment_drift": {"threshold": 0.10},
            "alerts": {"channels": [], "slack_webhook": None},
        }

        sent_drift = check_sentiment_drift(summary, params)
        topic_drift = check_topic_drift(summary, params)
        alerts = generate_alerts(sent_drift, topic_drift, params)

        assert sent_drift["drift_detected"] is True
        assert len(alerts) >= 1
        assert alerts[0]["type"] == "sentiment_drift"

    def test_monitoring_no_drift(self):
        summary = {
            "sentiment_percentages": {
                "positive": 34.0,
                "neutral": 33.0,
                "negative": 33.0,
            },
            "topic_distribution": {},
        }
        params = {
            "sentiment_drift": {"threshold": 0.10},
            "alerts": {"channels": []},
        }

        sent_drift = check_sentiment_drift(summary, params)
        topic_drift = check_topic_drift(summary, params)
        alerts = generate_alerts(sent_drift, topic_drift, params)

        assert sent_drift["drift_detected"] is False
        assert len(alerts) == 0


@pytest.mark.e2e
class TestUnifiedInsightsPipeline:
    """E2E: model outputs → unified interaction record."""

    def test_unified_record_generation(self):
        ids = [f"r{i}" for i in range(1, 4)]

        sentiment_df = pl.DataFrame({
            "review_id": ids,
            "customer_id": ["c1", "c2", "c3"],
            "product_id": ["p1", "p2", "p3"],
            "date": [datetime(2024, 1, i) for i in range(1, 4)],
            "sentiment": ["positive", "negative", "negative"],
            "sentiment_confidence": [0.95, 0.88, 0.60],
            "review_text": ["Great!", "Terrible!", "Meh shipping"],
        })
        aspects_df = pl.DataFrame({
            "review_id": ids,
            "aspect_sentiments": [
                [{"aspect": "quality", "sentiment": "positive", "confidence": 0.9}],
                [{"aspect": "price", "sentiment": "negative", "confidence": 0.8}],
                [],
            ],
        })
        topics_df = pl.DataFrame({
            "review_id": ids,
            "topic_name": ["Quality", "Pricing", "Shipping"],
        })
        emotions_df = pl.DataFrame({
            "review_id": ids,
            "detected_emotion": ["joy", "anger", "neutral"],
            "emotion_confidence": [0.9, 0.85, 0.5],
        })

        unified = create_unified_interaction_record(
            sentiment_df, aspects_df, topics_df, emotions_df,
        )

        assert len(unified) == 3
        assert "urgency" in unified.columns
        assert "recommended_action" in unified.columns
        assert "sample_output_record" in unified.columns

        urgencies = unified["urgency"].to_list()
        # r1: positive → Low
        assert urgencies[0] == "Low"
        # r2: negative + anger → High
        assert urgencies[1] == "High"


@pytest.mark.e2e
class TestCleaningPipeline:
    """E2E: dirty data → clean → validate → features."""

    def test_html_and_urls_removed_before_features(self):
        dirty = pl.DataFrame({
            "Interaction_ID": ["d1", "d2"],
            "Interaction_Payload": [
                "<p>Visit http://spam.com for deals!</p>",
                "<b>GREAT</b>   product   with   spaces",
            ],
            "Customer_ID": ["c1", "c2"],
            "Timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "Target_Object_ID": ["p1", "p2"],
            "Rating": [4.0, 5.0],
            "Channel_ID": ["Web_Review", "Web_Review"],
            "Language_Code": ["en", "en"],
        })

        cleaned = clean_reviews(dirty, {
            "remove_html": True,
            "remove_urls": True,
            "lowercase": True,
            "remove_extra_spaces": True,
        })

        texts = cleaned["Interaction_Payload"].to_list()
        for t in texts:
            assert "<" not in t
            assert "http" not in t
            assert t == t.lower()
            assert "  " not in t


@pytest.mark.e2e
class TestErrorPropagation:
    """E2E: validation errors propagate correctly."""

    def test_missing_required_columns_raises(self):
        bad_df = pl.DataFrame({
            "Interaction_ID": ["x1"],
            "some_other": ["text"],
            "Rating": [3.0],
            "Channel_ID": ["Web"],
            "Language_Code": ["en"],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_reviews(bad_df, {
                "required_columns": ["Interaction_ID", "Interaction_Payload"],
                "min_review_length": 1,
                "max_review_length": 10000,
                "rating_range": [1.0, 5.0],
                "max_null_percentage": 0.5,
            })
