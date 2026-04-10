"""Unit tests for ABSA (Aspect-Based Sentiment Analysis) pipeline nodes."""

import sys
from unittest.mock import MagicMock

# Stub out heavy optional dependencies that get pulled in transitively
# via data_science/__init__.py -> topic_modeling -> bertopic / hdbscan / groq.
_stubs = [
    "bertopic", "bertopic.representation",
    "hdbscan",
    "groq",
    "dotenv",
]
for _mod in _stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
import polars as pl
from typing import List, Dict, Any

from ai_core.pipelines.data_science.absa.nodes import (
    extract_aspects,
    _analyze_aspect_sentiment_lexicon,
    create_aspect_summary,
    DEFAULT_ASPECTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review_df(
    texts: list[str],
    *,
    text_col: str = "review_text",
    aspect_terms: list[list[str]] | None = None,
    aspect_sentiments: list[list[dict]] | None = None,
) -> pl.DataFrame:
    """Return a minimal DataFrame with review text and optional aspect columns."""
    data: dict[str, Any] = {
        text_col: texts,
    }
    if aspect_terms is not None:
        data["aspect_terms"] = aspect_terms
    if aspect_sentiments is not None:
        data["aspect_sentiments"] = aspect_sentiments
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# extract_aspects
# ---------------------------------------------------------------------------


class TestExtractAspects:
    """Tests for extract_aspects()."""

    def test_finds_correct_aspects_in_text(self):
        df = _make_review_df([
            "The price is great and the quality is amazing.",
            "Shipping was slow but packaging was good.",
        ])
        result = extract_aspects(df, {"predefined": DEFAULT_ASPECTS})

        aspects_col = result["aspect_terms"].to_list()
        assert "price" in aspects_col[0]
        assert "quality" in aspects_col[0]
        assert "shipping" in aspects_col[1]
        assert "packaging" in aspects_col[1]

    def test_no_aspects_found(self):
        df = _make_review_df([
            "This is a completely generic sentence with nothing specific.",
        ])
        result = extract_aspects(df, {"predefined": DEFAULT_ASPECTS})

        aspects_col = result["aspect_terms"].to_list()
        assert aspects_col[0] == []

    def test_custom_predefined_list(self):
        custom_aspects = ["battery", "screen", "camera"]
        df = _make_review_df([
            "The battery life is excellent and the screen is bright.",
            "Camera quality could be better.",
        ])
        result = extract_aspects(df, {"predefined": custom_aspects})

        aspects_col = result["aspect_terms"].to_list()
        assert "battery" in aspects_col[0]
        assert "screen" in aspects_col[0]
        assert "camera" not in aspects_col[0]
        assert "camera" in aspects_col[1]

    def test_case_insensitive_matching(self):
        df = _make_review_df(["The PRICE is too high and Quality is lacking."])
        result = extract_aspects(df, {"predefined": DEFAULT_ASPECTS})

        aspects_col = result["aspect_terms"].to_list()
        assert "price" in aspects_col[0]
        assert "quality" in aspects_col[0]

    def test_uses_interaction_payload_column(self):
        df = _make_review_df(
            ["Great price and fast delivery."],
            text_col="Interaction_Payload",
        )
        result = extract_aspects(df, {"predefined": DEFAULT_ASPECTS})

        aspects_col = result["aspect_terms"].to_list()
        assert "price" in aspects_col[0]
        assert "delivery" in aspects_col[0]

    def test_empty_text(self):
        df = _make_review_df([""])
        result = extract_aspects(df, {"predefined": DEFAULT_ASPECTS})

        aspects_col = result["aspect_terms"].to_list()
        assert aspects_col[0] == []

    def test_default_aspects_used_when_not_specified(self):
        df = _make_review_df(["The design is beautiful and the material feels premium."])
        result = extract_aspects(df, {})

        aspects_col = result["aspect_terms"].to_list()
        assert "design" in aspects_col[0]
        assert "material" in aspects_col[0]


# ---------------------------------------------------------------------------
# _analyze_aspect_sentiment_lexicon
# ---------------------------------------------------------------------------


class TestAnalyzeAspectSentimentLexicon:
    """Tests for _analyze_aspect_sentiment_lexicon()."""

    def test_positive_context(self):
        df = _make_review_df(
            ["The shipping was fast and great overall."],
            aspect_terms=[["shipping"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert len(sentiments[0]) == 1
        assert sentiments[0][0]["aspect"] == "shipping"
        assert sentiments[0][0]["sentiment"] == "positive"

    def test_negative_context(self):
        df = _make_review_df(
            ["The quality was terrible and I am disappointed."],
            aspect_terms=[["quality"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert len(sentiments[0]) == 1
        assert sentiments[0][0]["aspect"] == "quality"
        assert sentiments[0][0]["sentiment"] == "negative"

    def test_neutral_context(self):
        df = _make_review_df(
            ["The price is what it is."],
            aspect_terms=[["price"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert len(sentiments[0]) == 1
        assert sentiments[0][0]["aspect"] == "price"
        assert sentiments[0][0]["sentiment"] == "neutral"
        assert sentiments[0][0]["confidence"] == 0.5

    def test_multiple_aspects_mixed_sentiment(self):
        # Text must be long enough so 50-char context windows don't overlap.
        df = _make_review_df(
            [
                "The shipping was excellent and arrived super fast in two days. "
                "On the other hand the overall quality of the product was "
                "terrible and just awful in every way."
            ],
            aspect_terms=[["shipping", "quality"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert len(sentiments[0]) == 2
        aspect_map = {s["aspect"]: s["sentiment"] for s in sentiments[0]}
        assert aspect_map["shipping"] == "positive"
        assert aspect_map["quality"] == "negative"

    def test_empty_aspects(self):
        df = _make_review_df(
            ["Some review text."],
            aspect_terms=[[]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert sentiments[0] == []

    def test_empty_text(self):
        df = _make_review_df(
            [""],
            aspect_terms=[["price"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert sentiments[0] == []

    def test_confidence_bounded(self):
        df = _make_review_df(
            ["The shipping was excellent amazing perfect great wonderful awesome fantastic."],
            aspect_terms=[["shipping"]],
        )
        result = _analyze_aspect_sentiment_lexicon(df, "review_text")

        sentiments = result["aspect_sentiments"].to_list()
        assert sentiments[0][0]["confidence"] <= 0.95


# ---------------------------------------------------------------------------
# create_aspect_summary
# ---------------------------------------------------------------------------


class TestCreateAspectSummary:
    """Tests for create_aspect_summary()."""

    def _build_sentiments_df(
        self, aspect_sentiments: list[list[dict]]
    ) -> pl.DataFrame:
        return pl.DataFrame({"aspect_sentiments": aspect_sentiments})

    def test_basic_summary(self):
        sentiments = [
            [
                {"aspect": "price", "sentiment": "positive", "confidence": 0.9},
                {"aspect": "quality", "sentiment": "negative", "confidence": 0.8},
            ],
            [
                {"aspect": "price", "sentiment": "positive", "confidence": 0.85},
            ],
        ]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        assert result["total_aspects_analyzed"] == 3
        assert result["unique_aspects"] == 2
        assert "price" in result["aspects"]
        assert "quality" in result["aspects"]
        assert result["aspects"]["price"]["total_mentions"] == 2
        assert result["aspects"]["price"]["positive"] == 2

    def test_empty_data(self):
        df = self._build_sentiments_df([[]])
        result = create_aspect_summary(df, {})

        assert result["total_aspects_analyzed"] == 0
        assert result["aspects"] == {}

    def test_problem_aspects_detection(self):
        """Problem aspects: >30% negative and >=10 mentions."""
        bad_sentiments = [{"aspect": "shipping", "sentiment": "negative", "confidence": 0.8}] * 8
        ok_sentiments = [{"aspect": "shipping", "sentiment": "positive", "confidence": 0.8}] * 5
        all_sentiments = [bad_sentiments + ok_sentiments]

        df = self._build_sentiments_df(all_sentiments)
        result = create_aspect_summary(df, {})

        assert result["aspects"]["shipping"]["total_mentions"] == 13
        assert result["aspects"]["shipping"]["negative_pct"] > 30
        assert "shipping" in result["problem_aspects"]

    def test_problem_aspects_not_triggered_below_threshold(self):
        """Not flagged as problem if <10 mentions even with high negative %."""
        sentiments = [[
            {"aspect": "color", "sentiment": "negative", "confidence": 0.9},
            {"aspect": "color", "sentiment": "negative", "confidence": 0.9},
            {"aspect": "color", "sentiment": "positive", "confidence": 0.9},
        ]]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        assert "color" not in result["problem_aspects"]

    def test_strong_aspects_detection(self):
        """Strong aspects: >70% positive and >=10 mentions."""
        pos_sentiments = [{"aspect": "quality", "sentiment": "positive", "confidence": 0.9}] * 10
        neg_sentiments = [{"aspect": "quality", "sentiment": "negative", "confidence": 0.8}] * 2
        all_sentiments = [pos_sentiments + neg_sentiments]

        df = self._build_sentiments_df(all_sentiments)
        result = create_aspect_summary(df, {})

        assert result["aspects"]["quality"]["total_mentions"] == 12
        assert result["aspects"]["quality"]["positive_pct"] > 70
        assert "quality" in result["strong_aspects"]

    def test_strong_aspects_not_triggered_below_threshold(self):
        """Not flagged as strong if <10 mentions even with high positive %."""
        sentiments = [[
            {"aspect": "design", "sentiment": "positive", "confidence": 0.95},
            {"aspect": "design", "sentiment": "positive", "confidence": 0.90},
        ]]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        assert "design" not in result["strong_aspects"]

    def test_net_sentiment_calculation(self):
        sentiments = [[
            {"aspect": "price", "sentiment": "positive", "confidence": 0.9},
            {"aspect": "price", "sentiment": "negative", "confidence": 0.8},
            {"aspect": "price", "sentiment": "neutral", "confidence": 0.5},
        ]]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        # net_sentiment = (positive - negative) / total = (1 - 1) / 3 = 0.0
        assert result["aspects"]["price"]["net_sentiment"] == 0.0

    def test_sorted_by_mentions(self):
        sentiments = [[
            {"aspect": "price", "sentiment": "positive", "confidence": 0.9},
            {"aspect": "quality", "sentiment": "positive", "confidence": 0.9},
            {"aspect": "quality", "sentiment": "positive", "confidence": 0.9},
        ]]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        aspect_keys = list(result["aspects"].keys())
        assert aspect_keys[0] == "quality"

    def test_timestamp_present(self):
        sentiments = [[{"aspect": "price", "sentiment": "positive", "confidence": 0.9}]]
        df = self._build_sentiments_df(sentiments)
        result = create_aspect_summary(df, {})

        assert "timestamp" in result
