"""Feature View definitions for Churn forecasting Feature Store."""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Float64Vector, Int64, Int32

from .entities import customer_entity, subscription_entity

# --- RFM Features (Non-Contractual / BTYD) ---
rfm_features_source = FileSource(
    name="rfm_features_source",
    path="data/05_model_input/processed_data",
    timestamp_field="event_timestamp",
    file_format="delta",
)

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
    ],
    source=rfm_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "cltv", "model": "btyd"},
)

# --- Survival Features (Contractual) ---
survival_features_source = FileSource(
    name="survival_features_source",
    path="data/05_model_input/survival_data",
    timestamp_field="event_timestamp",
    file_format="delta",
)

survival_features_view = FeatureView(
    name="survival_features",
    entities=[subscription_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="T", dtype=Int64, description="Duration in time units"),
        Field(
            name="E", dtype=Int32, description="Event indicator (1=churned, 0=censored)"
        ),
    ],
    source=survival_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "cltv", "model": "survival"},
)

# --- Behavioral Features (XGBoost Refinement & Deep Learning) ---
behavioral_features_source = FileSource(
    name="behavioral_features_source",
    path="data/05_model_input/feature_store",
    timestamp_field="event_timestamp",
    file_format="delta",
)

behavioral_features_view = FeatureView(
    name="behavioral_features",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(
            name="total_engagement_value",
            dtype=Float64,
            description="Total engagement aggregation",
        ),
        Field(
            name="engagement_frequency",
            dtype=Int64,
            description="Number of engagement events",
        ),
        Field(name="active_days_count", dtype=Int64, description="Unique active days"),
        Field(
            name="engagement_sequence_flat",
            dtype=Float64Vector,
            description="Flattened time-series of engagement for deep learning models.",
        ),
    ],
    source=behavioral_features_source,
    online=True,
    tags={"team": "ai_core", "domain": "cltv", "model": "xgboost_dl"},
)
