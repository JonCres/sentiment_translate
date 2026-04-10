"""
Unit tests for marketing/voice-of-client data processing nodes.
"""

import sys
from unittest.mock import MagicMock

# Stub heavy transitive dependencies before any project import
sys.modules.setdefault("langdetect", MagicMock())
sys.modules.setdefault("deep_translator", MagicMock())
sys.modules.setdefault("presidio_analyzer", MagicMock())
sys.modules.setdefault("presidio_anonymizer", MagicMock())
sys.modules.setdefault("utils", MagicMock())
sys.modules.setdefault("utils.device", MagicMock())

import pandas as pd
import pytest

from ai_core.pipelines.data_processing.nodes import (
    map_to_skeleton,
    validate_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feedback():
    return pd.DataFrame({
        "id_col": ["1", "2", "3"],
        "text_col": ["Great service!", "Mal servicio", "Service moyen"],
        "ts_col": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "score_col": [9, 2, 5],
    })


@pytest.fixture
def mapping_params():
    return {
        "mandatory": {
            "Interaction_ID": "id_col",
            "Interaction_Payload": "text_col",
            "Timestamp": "ts_col",
            "NPS_Score": "score_col",
        },
        "optional": {},
        "defaults": {"Channel_ID": "Survey"},
    }


# ---------------------------------------------------------------------------
# map_to_skeleton
# ---------------------------------------------------------------------------

class TestMapToSkeleton:
    def test_maps_mandatory_columns(self, sample_feedback, mapping_params):
        result = map_to_skeleton(sample_feedback, mapping_params)
        assert "Interaction_ID" in result.columns
        assert "Interaction_Payload" in result.columns
        assert len(result) == 3

    def test_applies_defaults(self, sample_feedback, mapping_params):
        result = map_to_skeleton(sample_feedback, mapping_params)
        assert "Channel_ID" in result.columns
        assert result["Channel_ID"].iloc[0] == "Survey"

    def test_handles_missing_source_column_gracefully(self, sample_feedback):
        params = {
            "mandatory": {"Interaction_ID": "nonexistent_col"},
            "optional": {},
            "defaults": {"Interaction_ID": "default_id"},
        }
        result = map_to_skeleton(sample_feedback, params)
        assert "Interaction_ID" in result.columns
        assert result["Interaction_ID"].iloc[0] == "default_id"

    def test_optional_columns_mapped(self, sample_feedback):
        params = {
            "mandatory": {"Interaction_ID": "id_col"},
            "optional": {"Extra": "text_col"},
            "defaults": {},
        }
        result = map_to_skeleton(sample_feedback, params)
        assert "Extra" in result.columns


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    def test_valid_data_passes(self):
        df = pd.DataFrame({
            "Interaction_ID": ["1", "2"],
            "Interaction_Payload": ["hello", "world"],
        })
        result = validate_data(df, {})
        assert len(result) == 2

    def test_missing_mandatory_columns_raises(self):
        df = pd.DataFrame({"Other": [1]})
        with pytest.raises(ValueError, match="Missing mandatory skeleton columns"):
            validate_data(df, {})

    def test_empty_data_raises(self):
        df = pd.DataFrame({
            "Interaction_ID": pd.Series([], dtype=str),
            "Interaction_Payload": pd.Series([], dtype=str),
        })
        with pytest.raises(ValueError, match="Data is empty"):
            validate_data(df, {})

    def test_partial_mandatory_columns_raises(self):
        df = pd.DataFrame({"Interaction_ID": ["1"]})
        with pytest.raises(ValueError, match="Missing mandatory skeleton columns"):
            validate_data(df, {})
