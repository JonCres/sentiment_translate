# Kedro Configuration Directory

## 📑 Table of Contents

- [Directory Structure](#directory-structure)
- [Key Files](#key-files)
- [How Kedro Uses These Files](#how-kedro-uses-these-files)
- [Configuration Separation](#configuration-separation)
- [Environment-Specific Configuration](#environment-specific-configuration)
- [Best Practices](#best-practices)
- [Parameter Organization](#parameter-organization)
- [More Information](#more-information)

This directory contains **Kedro pipeline configuration** following standard Kedro conventions for the **Predictive CLTV Insights** project.

## Directory Structure

```
conf/
├── base/                    # Base configuration (version controlled)
│   ├── catalog.yml         # Data catalog definitions
│   ├── parameters.yml      # Pipeline parameters ← CORE LOGIC HERE
│   └── logging.yml         # Kedro logging configuration
└── local/                   # Local overrides (gitignored)
    └── credentials.yml      # Sensitive credentials (not in git)
```

## Key Files

### `base/parameters.yml` ⭐

**Purpose**: Centralized configuration for business logic, model hyperparameters, and pipeline execution settings.

**Contains**:

- **Skeleton**: Data mapping for different source schemas (Transactions, Subscriptions, Engagement).
- **Data Processing**: Cleaning rules, observation periods, and train/test split ratios.
- **AI Modeling**: Model types (lifetimes, xgboost) and hyperparameters for BG/NBD, Gamma-Gamma, and Weibull AFT.
- **Visualization**: Plot themes, directory paths, and specific plot toggles.
- **MLOps**: Experiment tracking (MLflow), metric definitions, and drift detection thresholds.

**Used by**: All Kedro nodes via `params:` syntax in pipeline definitions.

**Example structure**:

```yaml
skeleton:
  transactions:
    mapping:
      customer_id: "Customer ID"
      transaction_date: "InvoiceDate"
      transaction_value: "Price"

data_processing:
  observation_period_end: "2011-12-09"
  test_size: 0.2

modeling:
  model_type: "lifetimes"
  lifetimes:
    bg_nbd:
      penalizer_coef: 0.01
    gamma_gamma:
      penalizer_coef: 0.01

visualization:
  plot_theme: "seaborn-darkgrid"
  frequency_recency_matrix: true

mlops:
  experiment_name: "cltv_prediction"
  metrics: ["rmse", "mae", "cltv_mean"]
```

### `base/catalog.yml`

**Purpose**: Define all datasets (inputs/outputs) for Kedro pipelines.

**Contains**:

- Dataset definitions (CSV, Parquet, Delta Lake, etc.)
- Intermediate data locations in the `data/` directory
- Model artifacts (Pickle/Cloudpickle) locations
- Reporting output paths (Plots, JSON summaries)

### `base/logging.yml`

**Purpose**: Kedro logging configuration.

**Contains**:

- Log levels (INFO, DEBUG, ERROR)
- Log file locations and rotation settings
- Log formatting templates

### `local/` (gitignored)

**Purpose**: Environment-specific overrides and credentials.

**Use for**:

- Database credentials (host, port, user, password)
- API keys (e.g., MLflow tracking server, Groq API key)
- Local file paths that differ between developers
- Experiment-specific parameter overrides

**Example `local/credentials.yml` Template**:

```yaml
# DO NOT COMMIT THIS FILE
# Add conf/local/* to .gitignore

# MLflow Tracking Server
mlflow:
  tracking_uri: "http://localhost:5000"
  # For remote tracking:
  # tracking_uri: "https://mlflow.example.com"
  # username: "your_username"
  # password: "${oc.env:MLFLOW_PASSWORD}"  # From environment variable

# Groq API (for LLM interpretations)
groq:
  api_key: "${oc.env:GROQ_API_KEY}"  # Export GROQ_API_KEY=gsk_...

# Database Credentials (if using remote data)
database:
  host: "localhost"
  port: 5432
  username: "ai_core_user"
  password: "${oc.env:DB_PASSWORD}"
  database: "cltv_production"

# Feature Store (Feast)
feast:
  registry: "data/feast_registry.db"
  # For remote registry:
  # registry: "s3://feast-registry-bucket/registry.db"
  # aws_access_key_id: "${oc.env:AWS_ACCESS_KEY_ID}"
  # aws_secret_access_key: "${oc.env:AWS_SECRET_ACCESS_KEY}"

# Cloud Storage (if using S3/GCS)
storage:
  s3_bucket: "ai-core-artifacts"
  aws_region: "us-west-2"
  # Credentials from environment or IAM role
```

**Accessing Credentials in Nodes**:

Credentials are automatically merged with `parameters.yml` and accessible via the catalog or context:

```python
# In Kedro nodes - credentials injected automatically
def my_node(data: pl.DataFrame, credentials: Dict[str, Any]):
    mlflow_uri = credentials["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_uri)
```

---

## How Kedro Uses These Files

### 1. Parameters in Pipeline Definitions

```python
# In src/ai_core/pipelines/data_science/pipeline.py
node(
    func=train_bg_nbd_model,
    inputs=["processed_data", "params:modeling.lifetimes.bg_nbd"],  # ← Reads from parameters.yml
    outputs="bg_nbd_model",
)
```

### 2. Parameters in Node Functions

```python
# In src/ai_core/pipelines/data_science/nodes.py
def train_bg_nbd_model(data: pl.DataFrame, params: Dict[str, Any]) -> BetaGeoFitter:
    penalizer_coef = params.get("penalizer_coef", 0.0)  # From parameters.yml
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    # ...
```

---

## Configuration Separation

We maintain a clear separation between Kedro (Logic) and Prefect (Orchestration) configurations:

| Configuration Type | File                          | Purpose                                    |
| ------------------ | ----------------------------- | ------------------------------------------ |
| **Pipeline Logic** | `conf/base/parameters.yml`    | Data mappings, models, thresholds, MLOps   |
| **Orchestration**  | `configs/project_config.yaml` | Prefect deployments, schedules, work pools |

**DO NOT** put Prefect infrastructure settings in `parameters.yml`!

---

## Environment-Specific Configuration

Kedro supports environment-specific configuration via the `--env` flag:

```
conf/
├── base/           # Default configuration (used if no env specified)
├── local/          # Local overrides (gitignored)
├── dev/            # Development environment settings
├── staging/        # Staging environment settings
└── prod/           # Production environment settings
```

To run with a specific environment:

```bash
kedro run --env=prod
```

---

## Best Practices

### ✅ DO

- Put all model hyperparameters in `parameters.yml`
- Use `skeleton` mapping to handle different client data shapes
- Keep parameters organized by pipeline module
- Document complex parameter meanings with YAML comments

### ❌ DON'T

- Hardcode file paths or thresholds in node functions
- Commit passwords or API keys to `base/catalog.yml` or `base/parameters.yml`
- Reference orchestration settings (Prefect) within Kedro nodes

---

## Parameter Organization

The project organizes parameters into five main categories:

```yaml
# 1. Data Skeleton (Schema mapping)
skeleton:
  transactions: {...}
  subscriptions: {...}
  engagement: {...}

# 2. Data Processing (Cleaning & Splitting)
data_processing:
  observation_period_end: "..."
  test_size: 0.2

# 3. AI Modeling (Hyperparameters & Selection)
modeling:
  model_type: "lifetimes" # or xgboost
  lifetimes:
    bg_nbd: {...}
    gamma_gamma: {...}
  xgboost: {...}

# 4. Visualization (Plot settings)
visualization:
  plots_dir: "..."
  plot_theme: "..."

# 5. Monitoring (MLOps Tracking)
mlops:
  experiment_name: "..."
  drift_detection: {...}
```

---

---

## How to Add New Datasets to catalog.yml

### Step-by-Step Guide

#### 1. **Identify Data Layer**

Kedro organizes data into numbered layers following the Data Engineering Convention:

```
01_raw         → Raw data from source (unchanged)
02_intermediate → Partially processed data
03_primary     → Primary cleaned datasets
04_feature     → Feature engineering outputs
05_model_input → Final datasets for training
06_models      → Trained model artifacts
07_model_output → Predictions and scores
08_reporting   → Final reports, plots, dashboards
```

#### 2. **Choose Dataset Type**

Common dataset types in this project:

| Type | Use Case | Example |
|------|----------|---------|
| `pandas.CSVDataset` | Raw CSV files | Transaction data |
| `polars.CSVDataset` | CSV with Polars (faster) | Large transaction data |
| `PolarsParquetDataset` | Parquet with Polars | Intermediate processed data |
| `PolarsDeltaDataset` | Delta Lake tables | Feature store, versioned data |
| `pickle.PickleDataset` | Simple Python objects | Small models, metadata |
| `CloudpickleDataset` | Complex models (custom) | XGBoost, lifetimes models |
| `json.JSONDataset` | Configuration, metadata | Model performance metrics |

#### 3. **Add Dataset Definition**

**Example: Adding a new CSV input dataset**

```yaml
# conf/base/catalog.yml

# Input: Raw subscription data
raw_subscriptions:
  type: polars.CSVDataset
  filepath: data/01_raw/subscriptions.csv
  load_args:
    has_header: true
    infer_schema_length: 10000
  # Optional: Schema validation
  metadata:
    description: "Raw subscription data with start/end dates and status"
    schema:
      - subscription_id: String
      - customer_id: String
      - start_date: Date
      - end_date: Date (nullable)
      - status: String (active/churned/cancelled)
```

**Example: Adding Parquet intermediate dataset**

```yaml
# Intermediate: Cleaned subscriptions
cleaned_subscriptions:
  type: pandas.ParquetDataset  # or PolarsParquetDataset
  filepath: data/02_intermediate/cleaned_subscriptions.parquet
  save_args:
    compression: snappy
  metadata:
    description: "Subscriptions after cleaning and validation"
```

**Example: Adding Delta Lake feature store**

```yaml
# Feature store: Customer behavioral features
customer_features_delta:
  type: aicore.datasets.polars_delta_dataset.PolarsDeltaDataset
  filepath: data/05_model_input/customer_features
  save_args:
    mode: overwrite
    schema_mode: overwrite
  metadata:
    description: "Aggregated behavioral features per customer"
    primary_key: customer_id
    update_frequency: daily
```

**Example: Adding model artifact**

```yaml
# Model output: Trained XGBoost model
xgboost_churn_model:
  type: aicore.datasets.cloudpickle_dataset.CloudpickleDataset
  filepath: data/06_models/xgboost_churn.pkl
  backend: cloudpickle  # Handles complex models better than pickle
  versioned: true  # Enable versioning (creates timestamped subfolder)
  metadata:
    description: "XGBoost classifier for behavioral churn prediction"
    framework: xgboost
    framework_version: "2.0.3"
```

**Example: Adding reporting output**

```yaml
# Reporting: KPI summary JSON
cltv_kpi_summary:
  type: json.JSONDataset
  filepath: data/08_reporting/cltv_kpi_summary.json
  metadata:
    description: "Executive KPI summary (avg CLTV, churn rate, segments)"
```

#### 4. **Use Dataset in Pipeline**

```python
# In src/ai_core/pipelines/data_processing/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import clean_subscriptions

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=clean_subscriptions,
            inputs="raw_subscriptions",  # ← Matches catalog.yml key
            outputs="cleaned_subscriptions",  # ← Matches catalog.yml key
            name="clean_subscriptions_node"
        ),
    ])
```

#### 5. **Test Dataset Loading**

```bash
# Test that dataset can be loaded
kedro catalog list  # Show all datasets

kedro catalog info raw_subscriptions  # Show dataset details

# Load dataset interactively
kedro ipython
>>> catalog = context.catalog
>>> df = catalog.load("raw_subscriptions")
>>> print(df.head())
```

---

## How to Override Parameters Locally

### Use Case 1: Experiment with Hyperparameters

**Scenario**: You want to test different XGBoost parameters without modifying `base/parameters.yml`.

**Solution**: Create `conf/local/parameters.yml`:

```yaml
# conf/local/parameters.yml
# This file is gitignored - safe for experiments

modeling:
  xgboost:
    params:
      max_depth: 8  # Override: increased from 6
      learning_rate: 0.05  # Override: decreased from 0.1
      n_estimators: 200  # Override: increased from 100
```

**Result**: Kedro merges `local/parameters.yml` on top of `base/parameters.yml`. Only specified keys are overridden.

### Use Case 2: Use Different Data File Locally

**Scenario**: You have a local test dataset with different path.

**Solution**: Create `conf/local/catalog.yml`:

```yaml
# conf/local/catalog.yml
# Override dataset location for local testing

raw_data:
  type: polars.CSVDataset
  filepath: data/01_raw/test_sample_100k.csv  # Smaller test file
  # Inherits load_args from base/catalog.yml
```

### Use Case 3: Environment-Specific MLflow Tracking

**Scenario**: Use local MLflow server in development, remote in production.

**Solution**: Use environment variables with OmegaConf resolver:

```yaml
# conf/base/parameters.yml
mlops:
  mlflow_tracking_uri: "${oc.env:MLFLOW_TRACKING_URI,http://localhost:5000}"
  # Fallback to localhost if env var not set
```

Then set environment variable per environment:
```bash
# Development
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Production
export MLFLOW_TRACKING_URI="https://mlflow.production.example.com"
```

---

## MLflow Configuration Patterns

### Basic MLflow Setup

```yaml
# conf/base/parameters.yml

mlops:
  experiment_name: "cltv_prediction"
  run_name_prefix: "bg_nbd"

  # Tracking server (override in local/credentials.yml)
  tracking_uri: "http://localhost:5000"

  # Model registry
  model_registry_uri: "sqlite:///mlruns.db"

  # Autologging
  autolog:
    log_models: true
    log_datasets: true
    log_input_examples: true
    log_model_signatures: true

  # Metrics to track
  metrics:
    - name: "clv_12mo_mean"
      threshold: 400  # Alert if below
    - name: "clv_12mo_std"
      threshold: 100
    - name: "p_alive_mean"
      threshold: 0.6

  # Tags for run organization
  tags:
    team: "ml_ops"
    project: "cltv_insights"
    model_type: "probabilistic"
```

### MLflow Integration in Nodes

```python
# In src/ai_core/pipelines/data_science/nodes.py

import mlflow
from mlflow.models import infer_signature

def train_bg_nbd_model(data: pl.DataFrame, params: Dict[str, Any]) -> BetaGeoFitter:
    """Train BG/NBD model with MLflow tracking."""

    # MLflow experiment setup
    mlflow.set_experiment(params["mlops"]["experiment_name"])

    with mlflow.start_run(run_name=f"{params['mlops']['run_name_prefix']}_training"):
        # Log parameters
        mlflow.log_param("penalizer_coef", params["penalizer_coef"])
        mlflow.log_param("num_customers", len(data))

        # Train model
        bgf = BetaGeoFitter(penalizer_coef=params["penalizer_coef"])
        bgf.fit(data["frequency"], data["recency"], data["T"])

        # Log model parameters
        for param_name, param_value in bgf.params_.items():
            mlflow.log_metric(f"model_param_{param_name}", param_value)

        # Log model artifact
        mlflow.sklearn.log_model(bgf, "bg_nbd_model")

        # Log metrics
        mlflow.log_metric("training_samples", len(data))

        return bgf
```

---

## More Information

- [Kedro Configuration Documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html)
- [Kedro Parameters Documentation](https://docs.kedro.org/en/stable/configuration/parameters.html)
- [Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html)
- [OmegaConf Variable Interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
