"""Unit tests for monitoring pipeline nodes."""

import pytest
import polars as pl
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from ai_core.pipelines.monitoring.nodes import (
    check_sentiment_drift,
    check_topic_drift,
    generate_alerts,
    detect_high_urgency_alerts,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _monitoring_params(
    *,
    threshold: float = 0.15,
    channels: list[str] | None = None,
) -> Dict[str, Any]:
    """Return a standard monitoring_params dict."""
    return {
        "sentiment_drift": {"threshold": threshold},
        "alerts": {
            "channels": channels or [],
        },
    }


def _sentiment_summary(
    positive: float = 33.0,
    neutral: float = 33.0,
    negative: float = 34.0,
) -> Dict[str, Any]:
    return {
        "sentiment_percentages": {
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
        }
    }


def _topic_summary(topics: dict | None = None) -> Dict[str, Any]:
    return {
        "topic_distribution": topics or {"topic_a": 0.5, "topic_b": 0.5},
    }


# ---------------------------------------------------------------------------
# check_sentiment_drift
# ---------------------------------------------------------------------------


class TestCheckSentimentDrift:
    """Tests for check_sentiment_drift()."""

    @patch("ai_core.pipelines.monitoring.nodes.mlflow")
    def test_drift_detected_large_deviation(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None

        summary = _sentiment_summary(positive=80.0, neutral=10.0, negative=10.0)
        params = _monitoring_params(threshold=0.15)

        result = check_sentiment_drift(summary, params)

        assert result["drift_detected"] is True
        # Baseline is 33% each; 80 - 33 = 47 > threshold of 15
        assert result["max_drift"] > 15
        assert "current_distribution" in result

    @patch("ai_core.pipelines.monitoring.nodes.mlflow")
    def test_drift_not_detected_within_threshold(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None

        summary = _sentiment_summary(positive=35.0, neutral=32.0, negative=33.0)
        params = _monitoring_params(threshold=0.15)

        result = check_sentiment_drift(summary, params)

        assert result["drift_detected"] is False
        assert result["max_drift"] <= 15

    @patch("ai_core.pipelines.monitoring.nodes.mlflow")
    def test_mlflow_active_run_logs_metrics(self, mock_mlflow):
        mock_run = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        summary = _sentiment_summary(positive=60.0, neutral=20.0, negative=20.0)
        params = _monitoring_params(threshold=0.10)

        check_sentiment_drift(summary, params)

        mock_mlflow.log_metric.assert_any_call(
            "drift_max_sentiment_shift", pytest.approx(27.0, abs=1.0)
        )
        mock_mlflow.log_metric.assert_any_call("drift_detected", 1)

    @patch("ai_core.pipelines.monitoring.nodes.mlflow")
    def test_mlflow_inactive_run_no_logging(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None

        summary = _sentiment_summary()
        params = _monitoring_params()

        check_sentiment_drift(summary, params)

        mock_mlflow.log_metric.assert_not_called()

    @patch("ai_core.pipelines.monitoring.nodes.mlflow")
    def test_returns_timestamp(self, mock_mlflow):
        mock_mlflow.active_run.return_value = None

        summary = _sentiment_summary()
        params = _monitoring_params()

        result = check_sentiment_drift(summary, params)

        assert "timestamp" in result


# ---------------------------------------------------------------------------
# check_topic_drift
# ---------------------------------------------------------------------------


class TestCheckTopicDrift:
    """Tests for check_topic_drift()."""

    def test_placeholder_returns_no_drift(self):
        summary = _topic_summary()
        params = _monitoring_params()

        result = check_topic_drift(summary, params)

        assert result["drift_detected"] is False
        assert result["new_topics_count"] == 0

    def test_returns_timestamp(self):
        summary = _topic_summary({"returns": 0.3, "sizing": 0.7})
        params = _monitoring_params()

        result = check_topic_drift(summary, params)

        assert "timestamp" in result


# ---------------------------------------------------------------------------
# generate_alerts
# ---------------------------------------------------------------------------


class TestGenerateAlerts:
    """Tests for generate_alerts()."""

    def test_alert_generated_on_sentiment_drift(self):
        sentiment_drift = {
            "drift_detected": True,
            "max_drift": 25.0,
        }
        topic_drift = {
            "drift_detected": False,
        }
        params = _monitoring_params()

        alerts = generate_alerts(sentiment_drift, topic_drift, params)

        assert len(alerts) == 1
        assert alerts[0]["type"] == "sentiment_drift"
        assert alerts[0]["severity"] == "high"
        assert "25.00%" in alerts[0]["message"]

    def test_alert_generated_on_topic_drift(self):
        sentiment_drift = {"drift_detected": False}
        topic_drift = {"drift_detected": True}
        params = _monitoring_params()

        alerts = generate_alerts(sentiment_drift, topic_drift, params)

        assert len(alerts) == 1
        assert alerts[0]["type"] == "topic_drift"
        assert alerts[0]["severity"] == "medium"

    def test_alerts_generated_on_both_drifts(self):
        sentiment_drift = {"drift_detected": True, "max_drift": 20.0}
        topic_drift = {"drift_detected": True}
        params = _monitoring_params()

        alerts = generate_alerts(sentiment_drift, topic_drift, params)

        assert len(alerts) == 2
        types = {a["type"] for a in alerts}
        assert types == {"sentiment_drift", "topic_drift"}

    def test_no_alert_when_no_drift(self):
        sentiment_drift = {"drift_detected": False}
        topic_drift = {"drift_detected": False}
        params = _monitoring_params()

        alerts = generate_alerts(sentiment_drift, topic_drift, params)

        assert alerts == []

    def test_alert_has_timestamp(self):
        sentiment_drift = {"drift_detected": True, "max_drift": 30.0}
        topic_drift = {"drift_detected": False}
        params = _monitoring_params()

        alerts = generate_alerts(sentiment_drift, topic_drift, params)

        assert "timestamp" in alerts[0]


# ---------------------------------------------------------------------------
# detect_high_urgency_alerts
# ---------------------------------------------------------------------------


class TestDetectHighUrgencyAlerts:
    """Tests for detect_high_urgency_alerts()."""

    def test_detects_high_urgency_rows(self):
        df = pl.DataFrame({
            "Interaction_ID": ["id_1", "id_2", "id_3"],
            "customer_id": ["c1", "c2", "c3"],
            "urgency": ["High", "Low", "High"],
            "topic_name": ["Returns", "General", "Billing"],
            "detected_emotion": ["anger", "joy", "frustration"],
            "recommended_action": ["escalate", "none", "escalate"],
            "review_text": [
                "This is terrible, I want a refund immediately!",
                "Everything is fine, thanks.",
                "I've been waiting for weeks with no response!",
            ],
        })

        alerts = detect_high_urgency_alerts(df)

        assert len(alerts) == 2
        ids = {a["interaction_id"] for a in alerts}
        assert ids == {"id_1", "id_3"}

    def test_no_high_urgency(self):
        df = pl.DataFrame({
            "Interaction_ID": ["id_1", "id_2"],
            "customer_id": ["c1", "c2"],
            "urgency": ["Low", "Medium"],
            "topic_name": ["General", "Shipping"],
            "detected_emotion": ["joy", "neutral"],
            "recommended_action": ["none", "monitor"],
            "review_text": ["Good product.", "Acceptable speed."],
        })

        alerts = detect_high_urgency_alerts(df)

        assert alerts == []

    def test_alert_contains_expected_keys(self):
        df = pl.DataFrame({
            "Interaction_ID": ["id_1"],
            "customer_id": ["c1"],
            "urgency": ["High"],
            "topic_name": ["Returns"],
            "detected_emotion": ["anger"],
            "recommended_action": ["escalate"],
            "review_text": ["Terrible experience!"],
        })

        alerts = detect_high_urgency_alerts(df)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["interaction_id"] == "id_1"
        assert alert["customer_id"] == "c1"
        assert alert["topic"] == "Returns"
        assert alert["emotion"] == "anger"
        assert alert["action"] == "escalate"
        assert "timestamp" in alert

    def test_text_preview_truncated(self):
        long_text = "A" * 200
        df = pl.DataFrame({
            "Interaction_ID": ["id_1"],
            "customer_id": ["c1"],
            "urgency": ["High"],
            "topic_name": ["Billing"],
            "detected_emotion": ["anger"],
            "recommended_action": ["escalate"],
            "review_text": [long_text],
        })

        alerts = detect_high_urgency_alerts(df)

        assert alerts[0]["text_preview"].endswith("...")
        # 100 chars of text + "..."
        assert len(alerts[0]["text_preview"]) == 103

    def test_caps_at_50_alerts(self):
        n = 60
        df = pl.DataFrame({
            "Interaction_ID": [f"id_{i}" for i in range(n)],
            "customer_id": [f"c_{i}" for i in range(n)],
            "urgency": ["High"] * n,
            "topic_name": ["Returns"] * n,
            "detected_emotion": ["anger"] * n,
            "recommended_action": ["escalate"] * n,
            "review_text": ["Bad experience."] * n,
        })

        alerts = detect_high_urgency_alerts(df)

        assert len(alerts) == 50
