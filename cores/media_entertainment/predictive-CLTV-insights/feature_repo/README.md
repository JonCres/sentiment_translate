# Feast Feature Store Guide

**Project:** Predictive CLTV Insights
**Feature Store Version:** Feast 0.38+
**Backend:** Polars + Delta Lake
**Last Updated:** 2026-02-13

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Feature Definitions](#feature-definitions)
5. [Materialization](#materialization)
6. [Integration with Kedro](#integration-with-kedro)
7. [Adding New Features](#adding-new-features)
8. [Online vs Offline Serving](#online-vs-offline-serving)
9. [Common Operations](#common-operations)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Feast (Feature Store) centralizes feature management for CLTV prediction models, providing:

- **Centralized feature repository**: Single source of truth for all features
- **Point-in-time correct joins**: Prevents data leakage in training
- **Online/offline consistency**: Same features in training and inference
- **Feature versioning**: Track feature schema evolution
- **Feature discovery**: Explore available features via Feast UI

**Why Feast for CLTV?**
- Reuse RFM features across BG/NBD, Gamma-Gamma, and XGBoost models
- Maintain historical feature snapshots for reproducibility
- Enable real-time CLTV predictions via online serving
- Simplify feature engineering for new models

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kedro Data Pipeline                       │
│  (data_processing/nodes.py → RFM + Behavioral Features)     │
└────────────────────────┬────────────────────────────────────┘
                         │ Write Features
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Delta Lake Storage                          │
│  data/05_model_input/                                        │
│    ├── processed_data/       (RFM features)                  │
│    ├── survival_data/        (Survival features)             │
│    └── feature_store/        (Behavioral features)           │
└────────────────────────┬────────────────────────────────────┘
                         │ Register Features
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Feast Registry                             │
│  feature_repo/                                               │
│    ├── feature_store.yaml    (Feast config)                 │
│    ├── entities.py           (Entity definitions)           │
│    └── features.py           (Feature view definitions)     │
└────────────────────────┬────────────────────────────────────┘
                         │ Materialize & Serve
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Feature Serving (Online/Offline)                │
│  • Offline: Historical features for training                 │
│  • Online: Real-time features for inference (Redis/SQLite)  │
└─────────────────────────────────────────────────────────────┘
```

### Backend: Polars + Delta Lake

**Why Delta Lake?**
- ACID transactions for feature writes
- Time travel for point-in-time correct joins
- Schema evolution without breaking pipelines
- Scalable to billions of rows

**Why Polars?**
- 10x faster than Pandas for large datasets
- Native Arrow memory format
- Lazy evaluation for query optimization
- Seamless Delta Lake integration

---

## Quick Start

### 1. Initialize Feature Store

```bash
# Navigate to feature store directory
cd feature_repo

# Initialize Feast (creates feature_store.yaml if not exists)
feast init -t local

# Apply feature definitions to registry
feast apply

# Verify setup
feast feature-views list
feast entities list
```

### 2. Materialize Features (Offline → Online Store)

```bash
# Materialize all features for last 7 days
feast materialize-incremental $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S)

# Or materialize specific date range
feast materialize \
  --start-time 2024-01-01T00:00:00 \
  --end-time 2024-12-31T23:59:59
```

### 3. Fetch Features for Training

```python
from feast import FeatureStore
import polars as pl
from datetime import datetime

# Initialize feature store
store = FeatureStore(repo_path="feature_repo")

# Define entity DataFrame (customers to fetch features for)
entity_df = pl.DataFrame({
    "customer_id": ["C1", "C2", "C3"],
    "event_timestamp": [datetime(2024, 12, 1)] * 3
})

# Fetch historical features (offline store)
training_df = store.get_historical_features(
    entity_df=entity_df.to_pandas(),
    features=[
        "rfm_features:frequency",
        "rfm_features:recency",
        "rfm_features:monetary_value",
        "behavioral_features:watch_time",
        "behavioral_features:engagement_score"
    ]
).to_df()

print(training_df.head())
```

### 4. Fetch Features for Inference (Online Store)

```python
# Fetch real-time features for prediction
feature_vector = store.get_online_features(
    features=[
        "rfm_features:frequency",
        "behavioral_features:watch_time"
    ],
    entity_rows=[{"customer_id": "C1"}]
).to_dict()

print(feature_vector)
# Output: {'customer_id': ['C1'], 'frequency': [5], 'watch_time': [1200]}
```

---

## Feature Definitions

### Entities (`entities.py`)

**What are entities?**
Entities are the primary keys for feature joins (e.g., `customer_id`, `subscription_id`).

```python
from feast import Entity, ValueType

# Customer entity (for transactional CLTV models)
customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier for non-contractual settings (TVOD, F2P)"
)

# Subscription entity (for contractual CLTV models)
subscription = Entity(
    name="subscription_id",
    value_type=ValueType.STRING,
    description="Unique subscription identifier for contractual settings (SVOD, SaaS)"
)
```

### Feature Views (`features.py`)

**What are feature views?**
Feature views define collections of features from a single data source.

#### RFM Features (Transaction-based)

```python
from feast import FeatureView, Field
from feast.types import Float64, Int64
from datetime import timedelta

rfm_features = FeatureView(
    name="rfm_features",
    entities=["customer_id"],
    ttl=timedelta(days=90),  # Features valid for 90 days
    schema=[
        Field(name="frequency", dtype=Int64, description="Number of repeat purchases"),
        Field(name="recency", dtype=Float64, description="Days between first and last purchase"),
        Field(name="T", dtype=Float64, description="Customer age (days since first purchase)"),
        Field(name="monetary_value", dtype=Float64, description="Average transaction value (USD)")
    ],
    source="rfm_delta_source",  # Defined in feature_store.yaml
    online=True,  # Enable online serving
    tags={"team": "ml_ops", "model": "bg_nbd"}
)
```

#### Behavioral Features (Engagement-based)

```python
behavioral_features = FeatureView(
    name="behavioral_features",
    entities=["customer_id"],
    ttl=timedelta(days=30),  # More recent features
    schema=[
        Field(name="watch_time", dtype=Int64, description="Total watch time (minutes)"),
        Field(name="login_count", dtype=Int64, description="Number of login sessions"),
        Field(name="engagement_score", dtype=Float64, description="Composite engagement metric (0-1)"),
        Field(name="buffering_ratio", dtype=Float64, description="Buffering events / total playbacks"),
        Field(name="active_days_count", dtype=Int64, description="Number of unique active days")
    ],
    source="behavioral_delta_source",
    online=True,
    tags={"team": "ml_ops", "model": "xgboost_churn"}
)
```

#### Survival Features (Contractual settings)

```python
survival_features = FeatureView(
    name="survival_features",
    entities=["subscription_id"],
    ttl=timedelta(days=365),
    schema=[
        Field(name="T", dtype=Int64, description="Subscription duration (days)"),
        Field(name="E", dtype=Int64, description="Event indicator (1=churned, 0=active)"),
        Field(name="subscription_tier", dtype=Int64, description="Tier level (1=basic, 2=premium, 3=elite)")
    ],
    source="survival_delta_source",
    online=False,  # Offline-only (used for training)
    tags={"team": "ml_ops", "model": "weibull_aft"}
)
```

---

## Materialization

### What is Materialization?

Materialization copies features from **offline store** (Delta Lake) to **online store** (SQLite/Redis) for low-latency inference.

### Materialization Strategy

**Development (Local)**:
- Offline store: Delta Lake (local filesystem)
- Online store: SQLite (local file)
- Materialize on-demand: `feast materialize`

**Production**:
- Offline store: Delta Lake (S3/GCS)
- Online store: Redis cluster
- Automated materialization: Scheduled job (Airflow/Prefect)

### Automated Materialization with Prefect

```python
# src/prefect_orchestration/feature_materialization.py

from prefect import flow, task
from feast import FeatureStore
from datetime import datetime, timedelta

@task
def materialize_features(start_date: datetime, end_date: datetime):
    """Materialize features from offline to online store."""
    store = FeatureStore(repo_path="feature_repo")

    store.materialize(
        start_date=start_date,
        end_date=end_date
    )

    return f"Materialized features from {start_date} to {end_date}"

@flow(name="daily_feature_materialization")
def daily_materialization_flow():
    """Daily feature materialization flow."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)

    result = materialize_features(start_date, end_date)
    return result

# Schedule to run daily at 2 AM
if __name__ == "__main__":
    daily_materialization_flow.serve(
        name="feast-daily-materialization",
        cron="0 2 * * *"
    )
```

---

## Integration with Kedro

### Writing Features from Kedro Pipeline

```python
# src/ai_core/pipelines/data_processing/nodes.py

from feast import FeatureStore
import polars as pl
from datetime import datetime

def register_cltv_features_to_feast(
    processed_data: pl.DataFrame,
    survival_data: pl.DataFrame,
    feature_store_data: pl.DataFrame,
    feast_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Register CLTV features to Feast Feature Store."""

    # 1. Write features to Delta Lake (offline store)
    processed_data.write_delta(
        "data/05_model_input/processed_data",
        mode="overwrite"
    )

    behavioral_data.write_delta(
        "data/05_model_input/feature_store",
        mode="overwrite"
    )

    # 2. Apply Feast definitions (register feature views)
    store = FeatureStore(repo_path="feature_repo")
    store.apply([])  # Applies entities.py and features.py

    # 3. Materialize features (optional, can be done separately)
    # store.materialize_incremental(end_date=datetime.utcnow())

    return {"status": "success", "timestamp": datetime.now().isoformat()}
```

### Reading Features in Kedro Pipeline

```python
# src/ai_core/pipelines/data_science/nodes.py

from feast import FeatureStore
import polars as pl

def load_features_for_training(entity_ids: List[str]) -> pl.DataFrame:
    """Load features from Feast for model training."""

    store = FeatureStore(repo_path="feature_repo")

    # Create entity DataFrame
    entity_df = pl.DataFrame({
        "customer_id": entity_ids,
        "event_timestamp": [datetime(2024, 12, 1)] * len(entity_ids)
    })

    # Fetch historical features
    training_features = store.get_historical_features(
        entity_df=entity_df.to_pandas(),
        features=[
            "rfm_features:frequency",
            "rfm_features:recency",
            "rfm_features:T",
            "rfm_features:monetary_value",
            "behavioral_features:watch_time",
            "behavioral_features:engagement_score"
        ]
    ).to_df()

    return pl.from_pandas(training_features)
```

---

## Adding New Features

### Step-by-Step: Add "Content Genre Preference" Feature

#### 1. Update Data Processing Pipeline

```python
# src/ai_core/pipelines/data_processing/nodes.py

def extract_genre_preferences(engagement_data: pl.DataFrame) -> pl.DataFrame:
    """Aggregate content genre preferences per customer."""

    genre_prefs = engagement_data.group_by("customer_id").agg([
        pl.col("genre").filter(pl.col("genre") == "Action").count().alias("action_count"),
        pl.col("genre").filter(pl.col("genre") == "Drama").count().alias("drama_count"),
        pl.col("genre").filter(pl.col("genre") == "Comedy").count().alias("comedy_count"),
        pl.col("watch_time").sum().alias("total_watch_time")
    ]).with_columns([
        (pl.col("action_count") / pl.col("total_watch_time")).alias("action_preference"),
        (pl.col("drama_count") / pl.col("total_watch_time")).alias("drama_preference"),
        (pl.col("comedy_count") / pl.col("total_watch_time")).alias("comedy_preference")
    ])

    return genre_prefs
```

#### 2. Add to Feature View

```python
# feature_repo/features.py

from feast import FeatureView, Field
from feast.types import Float64

content_preference_features = FeatureView(
    name="content_preferences",
    entities=["customer_id"],
    ttl=timedelta(days=30),
    schema=[
        Field(name="action_preference", dtype=Float64, description="Proportion of action content consumed"),
        Field(name="drama_preference", dtype=Float64, description="Proportion of drama content consumed"),
        Field(name="comedy_preference", dtype=Float64, description="Proportion of comedy content consumed")
    ],
    source="content_prefs_delta_source",  # Define in feature_store.yaml
    online=True,
    tags={"team": "ml_ops", "model": "recommendation"}
)
```

#### 3. Define Data Source

```yaml
# feature_repo/feature_store.yaml

project: cltv_insights
registry: data/feast_registry.db
provider: local
online_store:
  type: sqlite
  path: data/feast_online.db

offline_store:
  type: file

entity_key_serialization_version: 2

# Add new source
sources:
  content_prefs_delta_source:
    type: delta
    path: data/05_model_input/content_preferences
    timestamp_field: event_timestamp
```

#### 4. Apply and Materialize

```bash
cd feature_repo

# Apply new feature view
feast apply

# Materialize features
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Verify
feast feature-views describe content_preferences
```

---

## Online vs Offline Serving

### Offline Serving (Training & Batch Inference)

**Use Case**: Fetch historical features for model training or batch predictions.

**Characteristics**:
- Reads from Delta Lake (offline store)
- Point-in-time correct joins (prevents data leakage)
- Supports large-scale batch operations
- Higher latency (seconds to minutes)

**Example**:
```python
store = FeatureStore(repo_path="feature_repo")

# Fetch features as of specific timestamps
entity_df = pl.DataFrame({
    "customer_id": ["C1", "C2"],
    "event_timestamp": [datetime(2024, 6, 1), datetime(2024, 7, 1)]
})

training_data = store.get_historical_features(
    entity_df=entity_df.to_pandas(),
    features=["rfm_features:frequency", "behavioral_features:watch_time"]
).to_df()
```

### Online Serving (Real-time Inference)

**Use Case**: Fetch latest features for real-time CLTV predictions in API.

**Characteristics**:
- Reads from Redis/SQLite (online store)
- Ultra-low latency (< 10ms)
- Only serves latest feature values
- Requires materialization first

**Example**:
```python
store = FeatureStore(repo_path="feature_repo")

# Fetch latest features for customer
features = store.get_online_features(
    features=["rfm_features:frequency", "behavioral_features:watch_time"],
    entity_rows=[{"customer_id": "C1"}]
).to_dict()

# Use features for prediction
clv_prediction = model.predict(features)
```

---

## Common Operations

### List All Features

```bash
feast feature-views list
```

### Describe Feature View

```bash
feast feature-views describe rfm_features
```

### Validate Feature Definitions

```bash
feast plan
```

### Inspect Feature Values

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Get sample of features
sample = store.get_historical_features(
    entity_df=pl.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "event_timestamp": [datetime.now()] * 3
    }).to_pandas(),
    features=["rfm_features:frequency", "rfm_features:recency"]
).to_df()

print(sample)
```

### Clear Online Store (Reset)

```bash
# SQLite (local)
rm data/feast_online.db

# Redis (production)
redis-cli FLUSHDB
```

---

## Troubleshooting

### Issue 1: `FeatureViewNotFoundException`

**Symptom**:
```
FeatureViewNotFoundException: Feature view 'rfm_features' not found
```

**Solution**:
```bash
cd feature_repo
feast apply  # Re-register feature views
```

### Issue 2: Materialization Fails

**Symptom**:
```
MaterializationError: No data found in offline store
```

**Solution**:
- Verify Delta Lake tables exist: `ls data/05_model_input/`
- Check feature view `source` matches Delta path in `feature_store.yaml`
- Ensure `event_timestamp` column exists in Delta tables

### Issue 3: Point-in-Time Join Returns Empty

**Symptom**:
`get_historical_features()` returns DataFrame with null feature values.

**Solution**:
- Ensure `event_timestamp` in entity_df matches feature timestamps
- Check TTL in feature view (features may have expired)
- Verify entity IDs exist in feature store

### Issue 4: Online Features Stale

**Symptom**:
`get_online_features()` returns outdated values.

**Solution**:
```bash
# Re-materialize recent data
feast materialize-incremental $(date -u -d '1 day ago' +%Y-%m-%dT%H:%M:%S)
```

---

## Additional Resources

- [Feast Documentation](https://docs.feast.dev/)
- [Delta Lake Python API](https://delta-io.github.io/delta-rs/python/)
- [Polars User Guide](https://docs.pola.rs/)
- Project-specific: `docs/technical_design.md` (Section 2.6: Feature Store Architecture)
