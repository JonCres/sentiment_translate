"""Feature View definitions for Voice of Customer Feature Store."""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from .entities import review_entity

# Source pointing to Delta table output from data_processing pipeline
review_features_source = FileSource(
    name="review_features_source",
    path="data/03_primary/review_features",
    timestamp_field="event_timestamp",
    file_format="delta",
)

# Feature View for review/VoC features
review_features_view = FeatureView(
    name="review_features",
    entities=[review_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="review_text", dtype=String, description="Cleaned review text"),
        Field(name="review_score", dtype=Float64, description="Review rating score"),
        Field(name="word_count", dtype=Int64, description="Number of words in review"),
        Field(
            name="char_count", dtype=Int64, description="Number of characters in review"
        ),
        Field(name="avg_word_length", dtype=Float64, description="Average word length"),
        Field(
            name="exclamation_count",
            dtype=Int64,
            description="Number of exclamation marks",
        ),
        Field(
            name="question_count", dtype=Int64, description="Number of question marks"
        ),
        Field(
            name="uppercase_ratio",
            dtype=Float64,
            description="Ratio of uppercase characters",
        ),
    ],
    source=review_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "voc", "industry": "retail_cpg"},
)
