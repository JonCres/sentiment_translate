# Kedro Configuration Directory

## 📑 Table of Contents

- [Directory Structure](#directory-structure)
- [Key Files](#key-files)
  - [`base/parameters.yml`](#baseparametersyml)
  - [`base/catalog.yml`](#basecatalogyml)
  - [`base/logging.yml`](#baseloggingyml)
  - [`local/`](#local-gitignored)
- [How Kedro Uses These Files](#how-kedro-uses-these-files)
- [Configuration Separation](#configuration-separation)
- [Environment-Specific Configuration](#environment-specific-configuration)
- [Best Practices](#best-practices)
- [Parameter Organization](#parameter-organization)
- [More Information](#more-information)

This directory contains **Kedro pipeline configuration** following standard Kedro conventions for the **Churn forecasting** project.

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

- **Skeleton**: Data mapping for raw source schemas (Transactions, Subscriptions, Engagement).
- **Data Processing**: Transformation logic, cleaning rules, and observation periods.
- **AI Modeling**: Model types, hyperparameters for XGBoost, Ensemble, and Deep Learning models.
- **MLOps & Feature Store**: Experiment tracking (MLflow), metric definitions (AUC-ROC, C-Index), drift detection, and Feast feature store paths.
- **Visualization**: Plot themes, directory paths, and specific plot toggles.

**Used by**: All Kedro nodes via `params:` syntax in pipeline definitions.

**Example structure**:

```yaml
# In conf/base/parameters.yml

# Section 1: Data Source Mapping
skeleton:
  transactions:
    mapping:
      customer_id: "Customer ID"
      transaction_date: "InvoiceDate"
      transaction_value: "Price"
      quantity: "Quantity"
    date_format: "%m/%d/%Y %H:%M"
  subscriptions:
    mapping:
      customer_id: "user_id"
      start_date: "sub_start"
      end_date: "sub_end"
    date_format: "%Y-%m-%d"
  engagement:
    mapping:
      customer_id: "client_id"
      date: "event_date"
      engagement_metric: "metric_name"
      engagement_value: "metric_value"

# Section 2: Data Processing Logic
data_processing:
  observation_period_end: "2011-12-09"
  inactivity_threshold_days: 90
  test_size: 0.2 # Used in data_science pipeline

# Section 3: AI Modeling Hyperparameters
modeling:
  model_type: "ensemble" # or "xgboost" or "deep_learning"
  deep_learning:
    sequence_length: 90
    lstm_units: 64
  xgboost:
    n_estimators: 100
    max_depth: 6

# Section 4: MLOps & Feature Store
mlops:
  experiment_name: "media_entertainment_churn_forecasting"
  metrics: ["auc_roc", "f1_score", "accuracy"]
feast:
  feature_repo_path: "feature_repo"
  rfm_delta_path: "data/05_model_input/processed_data"
  behavioral_delta_path: "data/05_model_input/feature_store"

# Section 5: Visualization
visualization:
  plot_theme: "seaborn-darkgrid"
  churn_distribution_plot: true
  risk_tier_plot: true
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
- API keys (e.g., MLflow tracking server)
- Local file paths that differ between developers

---

## How Kedro Uses These Files

### 1. Parameters in Pipeline Definitions

```python
# In src/ai_core/pipelines/data_science/pipeline.py
node(
    func=predict_churn,
    inputs=["ensemble_model", "feature_store"],
    outputs="raw_churn_predictions",
)
```

### 2. Parameters in Node Functions

```python
# In src/ai_core/pipelines/data_science/nodes.py
def predict_churn(ensemble_model: Any, feature_store: pl.DataFrame) -> pl.DataFrame:
    # Logic to calculate survival/hazard metrics
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
# 1. Data Skeleton (Schema mapping for raw sources)
skeleton:
  transactions:
    mapping: {...}
    date_format: "..."
  subscriptions:
    mapping: {...}
    date_format: "..."
    defaults: {...}
  engagement:
    mapping: {...}
    defaults: {...}

# 2. Data Processing (Transformation logic)
data_processing:
  observation_period_end: "..."
  inactivity_threshold_days: 90

# 3. AI Modeling (Hyperparameters & Selection)
modeling:
  model_type: "ensemble" # or "xgboost", "deep_learning"
  deep_learning:
    sequence_length: 90
  ensemble:
    weights: [0.4, 0.3, 0.2, 0.1]
  xgboost: {...}

# 4. MLOps (Tracking, Validation, and Feature Store)
mlops:
  experiment_name: "..."
  drift_detection: {...}
feast:
  feature_repo_path: "feature_repo"
  # ... other feast paths

# 5. Visualization (Plot settings)
visualization:
  plots_dir: "..."
  plot_theme: "..."
```

---

## More Information

- [Kedro Configuration Documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html)
- [Kedro Parameters Documentation](https://docs.kedro.org/en/stable/configuration/parameters.html)
- [Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html)
