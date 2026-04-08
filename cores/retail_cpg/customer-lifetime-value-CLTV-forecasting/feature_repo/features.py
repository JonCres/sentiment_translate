"""Feature View definitions for CLTV Forecasting Feature Store."""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, String

from .entities import customer_entity

# Source pointing to Delta table output from data_processing pipeline
rfm_features_source = FileSource(
    name="rfm_features_source",
    path="data/03_primary/processed_data",
    timestamp_field="event_timestamp",
    file_format="delta",
)

# Feature View for RFM features
rfm_features_view = FeatureView(
    name="rfm_features",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(
            name="frequency", dtype=Float64, description="Number of repeat purchases"
        ),
        Field(name="recency", dtype=Float64, description="Days since last purchase"),
        Field(name="T", dtype=Float64, description="Customer tenure in days"),
        Field(
            name="monetary_value",
            dtype=Float64,
            description="Average transaction value",
        ),
        Field(
            name="cohort_month", dtype=String, description="Customer cohort (YYYY-MM)"
        ),
        Field(
            name="acquisition_channel",
            dtype=String,
            description="Customer acquisition channel",
        ),
    ],
    source=rfm_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "cltv", "industry": "retail_cpg"},
)
