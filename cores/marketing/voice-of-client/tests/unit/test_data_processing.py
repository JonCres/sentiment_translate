import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the module under test
sys.modules["langdetect"] = MagicMock()
sys.modules["deep_translator"] = MagicMock()

import pandas as pd
import pytest
from unittest.mock import patch

# Now import the module
from ai_core.pipelines.data_processing.nodes import (
    validate_data, 
    clean_feedback_data, 
    detect_languages, 
    translate_feedback
)

@pytest.fixture
def sample_feedback_data():
    return pd.DataFrame({
        "Interaction_ID": ["1", "2", "3"],
        "Interaction_Payload": ["Great service", "Muy mal servicio", "Service moyen"],
        "Timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "NPS_Score": [9, 2, 5]
    })

def test_validate_data_success(sample_feedback_data):
    validated = validate_data(sample_feedback_data, {})
    pd.testing.assert_frame_equal(validated, sample_feedback_data)

def test_validate_data_missing_cols():
    df = pd.DataFrame({"Other_Col": [1]})
    with pytest.raises(ValueError, match="Missing mandatory skeleton columns"):
        validate_data(df, {})

def test_clean_feedback_data(sample_feedback_data):
    # Add a duplicate and a null score
    df = pd.concat([sample_feedback_data, sample_feedback_data.iloc[[0]]], ignore_index=True)
    df.loc[0, "NPS_Score"] = None
    
    cleaned = clean_feedback_data(df)
    
    # Check duplicates removed
    assert len(cleaned) == 3
    # Check string conversion
    assert cleaned["Interaction_Payload"].dtype == object
    # Check timestamp conversion
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Timestamp"])

def test_detect_languages_enabled(sample_feedback_data):
    config = {
        "language_detection": {
            "enabled": True,
            "fallback_lang": "en"
        }
    }
    
    # We need to patch the imported names in the module, 
    # since we mocked the modules themselves but the module under test 
    # might have imported specific names (from langdetect import detect)
    
    # Re-import to ensure our sys.modules hack is effective if it wasn't already loaded
    # But since we do 'from ... import ...', it should be fine if it's the first import.
    
    # However, 'from langdetect import detect' in nodes.py means 'detect' is a name in nodes.py
    # pointing to the mock's attribute.
    
    with patch("ai_core.pipelines.data_processing.nodes.detect") as mock_detect:
        mock_detect.side_effect = ["en", "es", "fr"]
        
        result = detect_languages(sample_feedback_data.copy(), config)
        
        assert "detected_language" in result.columns
        assert result.loc[0, "detected_language"] == "en"
        assert result.loc[1, "detected_language"] == "es"
        assert result.loc[2, "detected_language"] == "fr"

def test_translate_feedback_enabled(sample_feedback_data):
    config = {
        "language_detection": {
            "translation": {
                "enabled": True,
                "target_lang": "en"
            }
        }
    }
    
    df = sample_feedback_data.copy()
    df["detected_language"] = ["en", "es", "fr"]
    
    with patch("ai_core.pipelines.data_processing.nodes.GoogleTranslator") as mock_translator_cls:
        mock_instance = MagicMock()
        mock_translator_cls.return_value = mock_instance
        mock_instance.translate.side_effect = lambda x: f"Translated: {x}"
        
        result = translate_feedback(df, config)
        
        # English should not be translated
        assert result.loc[0, "Interaction_Payload"] == "Great service"
        # Spanish should be translated
        assert result.loc[1, "Interaction_Payload"] == "Translated: Muy mal servicio"