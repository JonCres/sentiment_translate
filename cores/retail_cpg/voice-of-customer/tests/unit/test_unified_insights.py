"""Tests for unified_insights pipeline nodes."""

import sys
from unittest.mock import MagicMock

# Stub heavy optional dependencies before importing project modules
_stubs = [
    "bertopic", "bertopic.representation",
    "hdbscan",
    "groq",
    "dotenv",
]
for _mod in _stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import polars as pl
import pytest

from ai_core.pipelines.data_science.unified_insights.nodes import (
    create_customer_aspect_profiles,
    create_unified_interaction_record,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_sentiment_df(
    *,
    id_col: str = "review_id",
    ids: list | None = None,
    sentiments: list | None = None,
    confidences: list | None = None,
) -> pl.DataFrame:
    ids = ids or ["r1", "r2", "r3"]
    sentiments = sentiments or ["positive", "negative", "negative"]
    confidences = confidences or [0.95, 0.85, 0.60]
    return pl.DataFrame({
        id_col: ids,
        "customer_id": [f"c{i}" for i in range(1, len(ids) + 1)],
        "product_id": [f"p{i}" for i in range(1, len(ids) + 1)],
        "date": ["2024-01-01"] * len(ids),
        "sentiment": sentiments,
        "sentiment_confidence": confidences,
        "review_text": [f"Review text {i}" for i in range(1, len(ids) + 1)],
    })


def _make_aspects_df(
    *,
    id_col: str = "review_id",
    ids: list | None = None,
    aspect_sentiments: list | None = None,
) -> pl.DataFrame:
    ids = ids or ["r1", "r2", "r3"]
    if aspect_sentiments is None:
        _defaults = [
            [{"aspect": "Quality", "sentiment": "positive", "confidence": 0.9}],
            [{"aspect": "Shipping", "sentiment": "negative", "confidence": 0.8}],
            [{"aspect": "Price", "sentiment": "neutral", "confidence": 0.7}],
        ]
        aspect_sentiments = _defaults[: len(ids)]
    return pl.DataFrame({
        id_col: ids,
        "customer_id": [f"c{i}" for i in range(1, len(ids) + 1)],
        "aspect_sentiments": aspect_sentiments,
    })


def _make_topics_df(
    *,
    id_col: str = "review_id",
    ids: list | None = None,
    topics: list | None = None,
) -> pl.DataFrame:
    ids = ids or ["r1", "r2", "r3"]
    if topics is None:
        _defaults = ["Quality", "Shipping", "Price"]
        topics = _defaults[: len(ids)]
    return pl.DataFrame({id_col: ids, "topic_name": topics})


def _make_emotions_df(
    *,
    id_col: str = "review_id",
    ids: list | None = None,
    emotions: list | None = None,
    confidences: list | None = None,
) -> pl.DataFrame:
    ids = ids or ["r1", "r2", "r3"]
    if emotions is None:
        _defaults = ["joy", "anger", "sadness"]
        emotions = _defaults[: len(ids)]
    if confidences is None:
        _defaults = [0.9, 0.85, 0.7]
        confidences = _defaults[: len(ids)]
    return pl.DataFrame({
        id_col: ids,
        "detected_emotion": emotions,
        "emotion_confidence": confidences,
    })


# ---------------------------------------------------------------------------
# Tests: create_unified_interaction_record
# ---------------------------------------------------------------------------

class TestCreateUnifiedInteractionRecord:
    """Tests for create_unified_interaction_record node."""

    def test_joins_all_inputs_on_review_id(self):
        """Unified DF contains columns from every input."""
        result = create_unified_interaction_record(
            _make_sentiment_df(),
            _make_aspects_df(),
            _make_topics_df(),
            _make_emotions_df(),
        )
        assert len(result) == 3
        for col in [
            "sentiment",
            "aspect_sentiments",
            "topic_name",
            "detected_emotion",
            "urgency",
            "recommended_action",
            "sample_output_record",
        ]:
            assert col in result.columns

    def test_joins_on_interaction_id_when_no_review_id(self):
        """Falls back to Interaction_ID when review_id is absent."""
        result = create_unified_interaction_record(
            _make_sentiment_df(id_col="Interaction_ID"),
            _make_aspects_df(id_col="Interaction_ID"),
            _make_topics_df(id_col="Interaction_ID"),
            _make_emotions_df(id_col="Interaction_ID"),
        )
        assert len(result) == 3
        assert "Interaction_ID" in result.columns

    # -- Urgency ----------------------------------------------------------

    def test_urgency_high_negative_high_confidence(self):
        """High urgency when sentiment is negative AND confidence > 0.8."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["negative"],
                confidences=[0.95],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"]),
            _make_emotions_df(ids=["r1"], emotions=["sadness"]),
        )
        assert result["urgency"].to_list() == ["High"]

    def test_urgency_high_anger_emotion(self):
        """High urgency when detected_emotion is anger."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["neutral"],
                confidences=[0.5],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"]),
            _make_emotions_df(ids=["r1"], emotions=["anger"]),
        )
        assert result["urgency"].to_list() == ["High"]

    def test_urgency_high_fear_emotion(self):
        """High urgency when detected_emotion is fear."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["positive"],
                confidences=[0.5],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"]),
            _make_emotions_df(ids=["r1"], emotions=["fear"]),
        )
        assert result["urgency"].to_list() == ["High"]

    def test_urgency_medium_negative_low_confidence(self):
        """Medium urgency when sentiment is negative but confidence <= 0.8."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["negative"],
                confidences=[0.60],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"]),
            _make_emotions_df(ids=["r1"], emotions=["sadness"]),
        )
        assert result["urgency"].to_list() == ["Medium"]

    def test_urgency_low_positive_sentiment(self):
        """Low urgency for positive sentiment with non-alarming emotion."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["positive"],
                confidences=[0.95],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"]),
            _make_emotions_df(ids=["r1"], emotions=["joy"]),
        )
        assert result["urgency"].to_list() == ["Low"]

    # -- Recommended actions -----------------------------------------------

    def test_action_positive_sentiment(self):
        """Positive sentiment → thank-you / loyalty action."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["positive"],
                confidences=[0.9],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"], topics=["Quality"]),
            _make_emotions_df(ids=["r1"], emotions=["joy"]),
        )
        assert result["recommended_action"].to_list() == [
            "Send Thank You Note / Promote Loyalty"
        ]

    def test_action_anger_emotion(self):
        """Anger emotion → crisis management action."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["negative"],
                confidences=[0.9],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"], topics=["General"]),
            _make_emotions_df(ids=["r1"], emotions=["anger"]),
        )
        assert result["recommended_action"].to_list() == [
            "Immediate Crisis Management Outreach"
        ]

    def test_action_shipping_negative(self):
        """Negative sentiment + Shipping topic → logistics action."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["negative"],
                confidences=[0.9],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"], topics=["Shipping"]),
            _make_emotions_df(ids=["r1"], emotions=["sadness"]),
        )
        assert result["recommended_action"].to_list() == [
            "Check Logistics / Delivery Status"
        ]

    def test_action_fallback_monitor(self):
        """Negative sentiment with unmatched topic falls back to monitor."""
        result = create_unified_interaction_record(
            _make_sentiment_df(
                ids=["r1"],
                sentiments=["negative"],
                confidences=[0.9],
            ),
            _make_aspects_df(ids=["r1"]),
            _make_topics_df(ids=["r1"], topics=["Uncategorized"]),
            _make_emotions_df(ids=["r1"], emotions=["sadness"]),
        )
        assert result["recommended_action"].to_list() == [
            "Monitor and Categorize"
        ]

    def test_sample_output_record_is_struct(self):
        """sample_output_record column is a Polars Struct type."""
        result = create_unified_interaction_record(
            _make_sentiment_df(),
            _make_aspects_df(),
            _make_topics_df(),
            _make_emotions_df(),
        )
        assert result["sample_output_record"].dtype == pl.Struct


# ---------------------------------------------------------------------------
# Tests: create_customer_aspect_profiles
# ---------------------------------------------------------------------------

class TestCreateCustomerAspectProfiles:
    """Tests for create_customer_aspect_profiles node."""

    def test_basic_aggregation(self):
        """Profiles contain one row per (customer_id, aspect) pair."""
        df = _make_aspects_df(
            ids=["r1", "r2"],
            aspect_sentiments=[
                [
                    {"aspect": "Quality", "sentiment": "positive", "confidence": 0.9},
                    {"aspect": "Price", "sentiment": "negative", "confidence": 0.8},
                ],
                [
                    {"aspect": "Quality", "sentiment": "negative", "confidence": 0.7},
                ],
            ],
        )
        profiles = create_customer_aspect_profiles(df)
        assert len(profiles) == 3  # c1-Quality, c1-Price, c2-Quality

    def test_sentiment_scores(self):
        """Scores: positive=1, neutral=0, negative=-1."""
        df = pl.DataFrame({
            "customer_id": ["c1", "c1"],
            "review_id": ["r1", "r2"],
            "aspect_sentiments": [
                [{"aspect": "Quality", "sentiment": "positive", "confidence": 0.9}],
                [{"aspect": "Quality", "sentiment": "negative", "confidence": 0.8}],
            ],
        })
        profiles = create_customer_aspect_profiles(df)
        row = profiles.filter(
            (pl.col("customer_id") == "c1") & (pl.col("aspect") == "Quality")
        )
        assert row["mention_count"].to_list() == [2]
        avg = row["avg_sentiment_score"].to_list()[0]
        assert avg == pytest.approx(0.0)  # (1 + -1) / 2

    def test_multiple_customers(self):
        """Each customer gets independent profile rows."""
        df = pl.DataFrame({
            "customer_id": ["c1", "c2"],
            "review_id": ["r1", "r2"],
            "aspect_sentiments": [
                [{"aspect": "Quality", "sentiment": "positive", "confidence": 0.9}],
                [{"aspect": "Quality", "sentiment": "negative", "confidence": 0.8}],
            ],
        })
        profiles = create_customer_aspect_profiles(df)
        assert profiles["customer_id"].n_unique() == 2

    def test_empty_aspect_sentiments(self):
        """Rows with empty aspect_sentiments list produce no profile rows."""
        df = pl.DataFrame({
            "customer_id": ["c1", "c2"],
            "review_id": ["r1", "r2"],
            "aspect_sentiments": [
                [],
                [{"aspect": "Quality", "sentiment": "positive", "confidence": 0.9}],
            ],
        })
        profiles = create_customer_aspect_profiles(df)
        # Only c2's non-empty aspects should survive
        assert len(profiles) == 1

    def test_null_aspect_sentiments_filtered(self):
        """Null aspect_sentiments are safely filtered out."""
        df = pl.DataFrame({
            "customer_id": ["c1", "c2"],
            "review_id": ["r1", "r2"],
            "aspect_sentiments": [
                None,
                [{"aspect": "Quality", "sentiment": "positive", "confidence": 0.9}],
            ],
        })
        profiles = create_customer_aspect_profiles(df)
        assert profiles["customer_id"].to_list() == ["c2"]

    def test_profile_columns(self):
        """Result contains expected aggregation columns."""
        df = _make_aspects_df(ids=["r1"])
        profiles = create_customer_aspect_profiles(df)
        for col in ["customer_id", "aspect", "mention_count", "avg_sentiment_score", "avg_confidence"]:
            assert col in profiles.columns
