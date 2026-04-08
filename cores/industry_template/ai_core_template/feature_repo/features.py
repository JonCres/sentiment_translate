"""Feature View definitions for the AI Core Template Feature Store."""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64

from .entities import customer_entity

# Source pointing to Delta table output from data_processing pipeline
customer_features_source = FileSource(
    name="customer_features_source",
    path="data/04_feature/features_delta",
    timestamp_field="event_timestamp",
    file_format="delta",
)

# Feature View for engineered customer features
customer_features_view = FeatureView(
    name="customer_features",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="clv", dtype=Float64, description="Customer Lifetime Value"),
        Field(name="purchase_freq", dtype=Float64, description="Purchase Frequency"),
        Field(name="total_spent", dtype=Float64, description="Total Amount Spent"),
        Field(name="months_active", dtype=Int64, description="Months Active"),
    ],
    source=customer_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "customer"},
)
