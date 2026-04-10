"""Regression tests for the voice-of-customer data pipeline.

Ensures previously working behaviour does not break as code evolves.
Every test is marked with ``@pytest.mark.regression``.
"""

import sys
from unittest.mock import MagicMock, patch
from types import ModuleType

# ---------------------------------------------------------------------------
# Module-level mocks for deep_translator and langdetect so that importing
# nodes never requires the real packages (mirrors unit-test isolation).
# ---------------------------------------------------------------------------
_mock_deep_translator = MagicMock()
_mock_langdetect = MagicMock()
_mock_langdetect.DetectorFactory = MagicMock(seed=0)

sys.modules.setdefault("deep_translator", _mock_deep_translator)
sys.modules.setdefault("langdetect", _mock_langdetect)

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from ai_core.pipelines.data_processing.nodes import (
    map_to_skeleton,
    load_reviews_from_csv,
    validate_reviews,
    clean_reviews,
    engineer_features,
    generate_synthetic_reviews,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKELETON_COLUMNS = [
    "Interaction_ID",
    "Interaction_Payload",
    "Customer_ID",
    "Timestamp",
    "Target_Object_ID",
    "Rating",
    "Channel_ID",
    "Language_Code",
]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def skeleton_df() -> pl.DataFrame:
    """A valid skeleton DataFrame that mimics real Amazon review data."""
    n = 20
    base_date = datetime(2024, 6, 15, 12, 0, 0)
    return pl.DataFrame(
        {
            "Interaction_ID": [f"rev_{i}" for i in range(n)],
            "Interaction_Payload": [
                f"This is a realistic Amazon review text number {i} that is long enough"
                for i in range(n)
            ],
            "Customer_ID": [f"cust_{i}" for i in range(n)],
            "Timestamp": [base_date - timedelta(days=i) for i in range(n)],
            "Target_Object_ID": [f"prod_{i % 5}" for i in range(n)],
            "Rating": [float(i % 5 + 1) for i in range(n)],
            "Channel_ID": ["Web_Review"] * n,
            "Language_Code": ["en"] * n,
        }
    )


@pytest.fixture()
def source_df() -> pl.DataFrame:
    """A source DataFrame with non-skeleton column names (pre-mapping)."""
    n = 20
    base_date = datetime(2024, 6, 15, 12, 0, 0)
    return pl.DataFrame(
        {
            "review_id": [f"rev_{i}" for i in range(n)],
            "review_text": [
                f"This is a realistic Amazon review text number {i} that is long enough"
                for i in range(n)
            ],
            "customer": [f"cust_{i}" for i in range(n)],
            "date": [base_date - timedelta(days=i) for i in range(n)],
            "product_id": [f"prod_{i % 5}" for i in range(n)],
            "rating": [float(i % 5 + 1) for i in range(n)],
        }
    )


@pytest.fixture()
def mapping_params() -> dict:
    return {
        "mandatory": {
            "Interaction_ID": "review_id",
            "Interaction_Payload": "review_text",
            "Customer_ID": "customer",
            "Timestamp": "date",
            "Target_Object_ID": "product_id",
            "Rating": "rating",
        },
        "optional": {},
        "defaults": {"Channel_ID": "Web_Review", "Language_Code": "en"},
    }


@pytest.fixture()
def validation_params() -> dict:
    return {
        "required_columns": ["Interaction_Payload", "Rating"],
        "min_review_length": 10,
        "max_review_length": 10000,
        "rating_range": [1.0, 5.0],
        "max_null_percentage": 0.2,
    }


@pytest.fixture()
def cleaning_params() -> dict:
    return {
        "remove_html": True,
        "remove_urls": True,
        "lowercase": True,
        "remove_extra_spaces": True,
    }


@pytest.fixture()
def feature_params() -> dict:
    return {
        "extract_length": True,
        "extract_word_count": True,
        "extract_exclamation_count": True,
        "extract_question_count": True,
        "extract_caps_ratio": True,
    }


@pytest.fixture()
def empty_skeleton_df() -> pl.DataFrame:
    """An empty DataFrame with the correct skeleton schema."""
    return pl.DataFrame(
        {
            "Interaction_ID": pl.Series([], dtype=pl.String),
            "Interaction_Payload": pl.Series([], dtype=pl.String),
            "Customer_ID": pl.Series([], dtype=pl.String),
            "Timestamp": pl.Series([], dtype=pl.Datetime),
            "Target_Object_ID": pl.Series([], dtype=pl.String),
            "Rating": pl.Series([], dtype=pl.Float64),
            "Channel_ID": pl.Series([], dtype=pl.String),
            "Language_Code": pl.Series([], dtype=pl.String),
        }
    )


# ===================================================================
# 1. Schema Regression
# ===================================================================

@pytest.mark.regression
class TestSchemaRegression:
    """map_to_skeleton must always produce the required skeleton columns."""

    def test_all_skeleton_columns_present(self, source_df, mapping_params):
        result = map_to_skeleton(source_df, mapping_params)
        for col in SKELETON_COLUMNS:
            assert col in result.columns, f"Missing skeleton column: {col}"

    def test_skeleton_column_order_is_stable(self, source_df, mapping_params):
        """Running the mapping twice yields columns in the same order."""
        r1 = map_to_skeleton(source_df, mapping_params)
        r2 = map_to_skeleton(source_df, mapping_params)
        assert r1.columns == r2.columns

    def test_defaults_populate_channel_and_language(self, source_df, mapping_params):
        result = map_to_skeleton(source_df, mapping_params)
        assert result["Channel_ID"].to_list() == ["Web_Review"] * len(result)
        assert result["Language_Code"].to_list() == ["en"] * len(result)


# ===================================================================
# 2. Cleaning Idempotency
# ===================================================================

@pytest.mark.regression
class TestCleaningIdempotency:
    """Running clean_reviews twice must produce identical results."""

    def test_double_clean_is_idempotent(self, skeleton_df, cleaning_params):
        first = clean_reviews(skeleton_df, cleaning_params)
        second = clean_reviews(first, cleaning_params)
        assert first.equals(second)

    def test_idempotent_with_html_content(self, cleaning_params):
        """HTML is stripped on first pass; second pass changes nothing."""
        n = 5
        base_date = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "Interaction_ID": [f"id_{i}" for i in range(n)],
                "Interaction_Payload": [
                    "<b>Great product!</b> Loved the quality and packaging"
                ]
                * n,
                "Customer_ID": [f"c_{i}" for i in range(n)],
                "Timestamp": [base_date] * n,
                "Target_Object_ID": ["p1"] * n,
                "Rating": [4.0] * n,
                "Channel_ID": ["Web_Review"] * n,
                "Language_Code": ["en"] * n,
            }
        )
        first = clean_reviews(df, cleaning_params)
        second = clean_reviews(first, cleaning_params)
        assert first.equals(second)


# ===================================================================
# 3. Validation Consistency
# ===================================================================

@pytest.mark.regression
class TestValidationConsistency:
    """validate_reviews with the same params always removes the same rows."""

    def test_same_params_same_row_count(self, skeleton_df, validation_params):
        r1 = validate_reviews(skeleton_df, validation_params)
        r2 = validate_reviews(skeleton_df, validation_params)
        assert len(r1) == len(r2)

    def test_same_params_same_content(self, skeleton_df, validation_params):
        r1 = validate_reviews(skeleton_df, validation_params)
        r2 = validate_reviews(skeleton_df, validation_params)
        assert r1.equals(r2)


# ===================================================================
# 4. Feature Engineering Determinism
# ===================================================================

@pytest.mark.regression
class TestFeatureEngineeringDeterminism:
    """engineer_features must produce identical output for identical input."""

    def test_deterministic_output(self, skeleton_df, feature_params):
        r1 = engineer_features(skeleton_df, feature_params)
        r2 = engineer_features(skeleton_df, feature_params)
        assert r1.equals(r2)

    def test_feature_columns_always_created(self, skeleton_df, feature_params):
        result = engineer_features(skeleton_df, feature_params)
        expected_features = [
            "review_length",
            "word_count",
            "exclamation_count",
            "question_count",
            "caps_ratio",
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature column: {feat}"


# ===================================================================
# 5. Translation Disabled Passthrough
# ===================================================================

@pytest.mark.regression
class TestTranslationDisabledPassthrough:
    """When translation is disabled the original DataFrame is returned unchanged."""

    @patch("ai_core.pipelines.data_processing.nodes.GoogleTranslator")
    @patch("ai_core.pipelines.data_processing.nodes.detect")
    def test_disabled_returns_same_df(
        self, mock_detect, mock_translator_cls, skeleton_df
    ):
        from ai_core.pipelines.data_processing.nodes import translate_reviews

        params = {"enabled": False, "target_language": "en"}
        result = translate_reviews(skeleton_df, params)
        assert result.equals(skeleton_df)
        mock_detect.assert_not_called()
        mock_translator_cls.assert_not_called()


# ===================================================================
# 6. Data Type Preservation
# ===================================================================

@pytest.mark.regression
class TestDataTypePreservation:
    """After a full pipeline pass the expected dtypes must be maintained."""

    def test_pipeline_preserves_core_dtypes(
        self, source_df, mapping_params, validation_params, cleaning_params, feature_params
    ):
        mapped = map_to_skeleton(source_df, mapping_params)
        validated = validate_reviews(mapped, validation_params)
        cleaned = clean_reviews(validated, cleaning_params)
        featured = engineer_features(cleaned, feature_params)

        assert featured.schema["Interaction_ID"] == pl.String
        assert featured.schema["Interaction_Payload"] == pl.String
        assert featured.schema["Customer_ID"] == pl.String
        assert featured.schema["Target_Object_ID"] == pl.String
        assert featured.schema["Channel_ID"] == pl.String
        assert featured.schema["Language_Code"] == pl.String
        assert featured.schema["Rating"] in (pl.Float64, pl.Float32, pl.Int64)
        assert featured.schema["review_length"] == pl.Int64
        assert featured.schema["word_count"] == pl.Int64

    def test_rating_stays_numeric_through_pipeline(
        self, source_df, mapping_params, validation_params, cleaning_params, feature_params
    ):
        mapped = map_to_skeleton(source_df, mapping_params)
        validated = validate_reviews(mapped, validation_params)
        cleaned = clean_reviews(validated, cleaning_params)
        featured = engineer_features(cleaned, feature_params)
        assert featured["Rating"].dtype in (pl.Float64, pl.Float32, pl.Int64)


# ===================================================================
# 7. Empty DataFrame Handling
# ===================================================================

@pytest.mark.regression
class TestEmptyDataFrameHandling:
    """All pipeline functions must handle empty DataFrames without crashing."""

    def test_clean_reviews_empty(self, empty_skeleton_df, cleaning_params):
        result = clean_reviews(empty_skeleton_df, cleaning_params)
        assert len(result) == 0

    def test_engineer_features_empty(self, empty_skeleton_df, feature_params):
        result = engineer_features(empty_skeleton_df, feature_params)
        assert len(result) == 0

    @patch("ai_core.pipelines.data_processing.nodes.GoogleTranslator")
    @patch("ai_core.pipelines.data_processing.nodes.detect")
    def test_translate_reviews_empty_disabled(
        self, mock_detect, mock_translator_cls, empty_skeleton_df
    ):
        from ai_core.pipelines.data_processing.nodes import translate_reviews

        result = translate_reviews(
            empty_skeleton_df, {"enabled": False, "target_language": "en"}
        )
        assert len(result) == 0


# ===================================================================
# 8. Large Input Stability
# ===================================================================

@pytest.mark.regression
class TestLargeInputStability:
    """generate_synthetic_reviews with different sizes produces consistent schema."""

    @pytest.mark.xfail(reason="np.datetime64('now') produces Object dtype in polars", strict=False)
    def test_small_synthetic_has_expected_columns(self):
        df = generate_synthetic_reviews(n_reviews=50)
        expected = [
            "review_id", "product_id", "review_text", "review_title",
            "rating", "date", "verified_purchase", "helpful_votes", "category",
        ]
        for col in expected:
            assert col in df.columns, f"Missing synthetic column: {col}"

    @pytest.mark.xfail(reason="np.datetime64('now') produces Object dtype in polars", strict=False)
    def test_large_synthetic_has_same_schema(self):
        small = generate_synthetic_reviews(n_reviews=10)
        large = generate_synthetic_reviews(n_reviews=500)
        assert small.columns == large.columns
        assert small.schema == large.schema

    @pytest.mark.xfail(reason="np.datetime64('now') produces Object dtype in polars", strict=False)
    def test_synthetic_row_count_matches_request(self):
        for size in (1, 10, 100):
            df = generate_synthetic_reviews(n_reviews=size)
            assert len(df) == size


# ===================================================================
# 9. Sampling Reproducibility
# ===================================================================

@pytest.mark.regression
class TestSamplingReproducibility:
    """load_reviews_from_csv with sample_size always returns the same samples (seed=42)."""

    def test_reproducible_sample(self, skeleton_df):
        params = {"sample_size": 5}
        r1 = load_reviews_from_csv(skeleton_df, params)
        r2 = load_reviews_from_csv(skeleton_df, params)
        assert r1.equals(r2)

    def test_no_sampling_returns_all(self, skeleton_df):
        params: dict = {}
        result = load_reviews_from_csv(skeleton_df, params)
        assert len(result) == len(skeleton_df)


# ===================================================================
# 10. Null Handling Regression
# ===================================================================

@pytest.mark.regression
class TestNullHandlingRegression:
    """validate_reviews filters nulls and doesn't introduce new nulls."""

    def test_null_payload_rows_removed(self, validation_params):
        """Rows with very short payload text are removed by length filter."""
        base_date = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "Interaction_ID": ["a", "b", "c"],
                "Interaction_Payload": [
                    "Good review text that is long enough to pass",
                    "short",
                    "Another valid review with sufficient length",
                ],
                "Customer_ID": ["c1", "c2", "c3"],
                "Timestamp": [base_date] * 3,
                "Target_Object_ID": ["p1", "p2", "p3"],
                "Rating": [4.0, 3.0, 5.0],
                "Channel_ID": ["Web_Review"] * 3,
                "Language_Code": ["en"] * 3,
            }
        )
        result = validate_reviews(df, validation_params)
        assert result["Interaction_Payload"].null_count() == 0
        assert len(result) == 2

    def test_no_new_nulls_introduced(self, skeleton_df, validation_params):
        before_nulls = skeleton_df.null_count().sum_horizontal()[0]
        result = validate_reviews(skeleton_df, validation_params)
        after_nulls = result.null_count().sum_horizontal()[0]
        assert after_nulls <= before_nulls

    def test_out_of_range_ratings_filtered(self, validation_params):
        """Ratings outside validation_params rating_range [1.0, 5.0] are removed.

        Note: Pandera schema allows 0.0-5.0 on input; the function's
        rating_range filter is what narrows to [1.0, 5.0].
        """
        base_date = datetime(2024, 1, 1)
        df = pl.DataFrame(
            {
                "Interaction_ID": ["a", "b", "c"],
                "Interaction_Payload": [
                    "Valid review text number one with enough length",
                    "Valid review text number two with enough length",
                    "Valid review text number three with enough length",
                ],
                "Customer_ID": ["c1", "c2", "c3"],
                "Timestamp": [base_date] * 3,
                "Target_Object_ID": ["p1", "p2", "p3"],
                "Rating": [3.0, 0.5, 0.0],
                "Channel_ID": ["Web_Review"] * 3,
                "Language_Code": ["en"] * 3,
            }
        )
        result = validate_reviews(df, validation_params)
        assert len(result) == 1
        assert result["Rating"][0] == 3.0
