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

This directory contains **Kedro pipeline configuration** following standard Kedro conventions.

## Directory Structure

```
conf/
├── base/                    # Base configuration (version controlled)
│   ├── catalog.yml         # Data catalog definitions
│   ├── parameters.yml      # Pipeline parameters ← PIPELINE LOGIC HERE
│   └── logging.yml         # Kedro logging configuration
└── local/                   # Local overrides (gitignored)
    └── credentials.yml      # Sensitive credentials (not in git)
```

## Key Files

### `base/parameters.yml` ⭐
**Purpose**: Pipeline logic parameters for all Kedro pipelines

**Contains**:
- Data processing parameters (data sources, validation rules, cleaning settings)
- Sentiment analysis model configuration
- Topic modeling settings
- Visualization parameters
- Monitoring thresholds and drift detection settings

**Used by**: All Kedro nodes via `params:` syntax in pipeline definitions

**Example structure**:
```yaml
data_processing:
  download:
    dataset: "shijli/amazon-reviews-multi"
    sample_size: 50000
  validation:
    min_review_length: 10
    required_columns: [...]
  cleaning:
    remove_html: true
    lowercase: true

sentiment_analysis:
  model:
    name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: 32

topic_modeling:
  model:
    algorithm: "BERTopic"
    nr_topics: "auto"

visualization:
  sentiment_dist:
    show_percentages: true

monitoring:
  sentiment_drift:
    threshold: 0.05
```

### `base/catalog.yml`
**Purpose**: Define all datasets (inputs/outputs) for Kedro pipelines

**Contains**:
- Dataset definitions (type, filepath, load/save args)
- Intermediate data locations
- Model artifact locations
- Visualization output paths

### `base/logging.yml`
**Purpose**: Kedro logging configuration

**Contains**:
- Log levels for different loggers
- Log file locations
- Log formatting

### `local/` (gitignored)
**Purpose**: Environment-specific overrides and credentials

**Use for**:
- Database credentials
- API keys
- Local file paths
- Development-specific settings

---

## How Kedro Uses These Files

### 1. Parameters in Pipeline Definitions

```python
# In src/aicore_voice_of_customer/pipelines/data_processing/pipeline.py
node(
    func=download_amazon_reviews,
    inputs="params:data_processing.download",  # ← Reads from parameters.yml
    outputs="amazon_reviews_raw",
)
```

### 2. Parameters in Node Functions

```python
# In src/aicore_voice_of_customer/pipelines/data_processing/nodes.py
def download_amazon_reviews(params: Dict[str, Any]) -> pd.DataFrame:
    dataset_name = params['dataset']  # From parameters.yml
    sample_size = params['sample_size']
    # ...
```

### 3. Accessing Nested Parameters

```python
# In pipeline definition
inputs=[
    "reviews_cleaned",
    "params:data_processing.features"  # ← Gets data_processing.features section
]

# In node function
def engineer_features(df: pd.DataFrame, feature_params: Dict[str, Any]):
    if feature_params['extract_length']:  # From parameters.yml
        df['length'] = df['text'].str.len()
```

---

## Configuration Separation

We maintain a clear separation between Kedro and Prefect configurations:

| Configuration Type | File                          | Purpose                                    |
| ------------------ | ----------------------------- | ------------------------------------------ |
| **Pipeline Logic** | `conf/base/parameters.yml`    | Data, models, features, thresholds         |
| **Orchestration**  | `configs/project_config.yaml` | Prefect deployments, schedules, work pools |

**DO NOT** put Prefect orchestration settings in `parameters.yml`!

---

## Environment-Specific Configuration

Kedro supports environment-specific configuration:

```
conf/
├── base/           # Default configuration
├── local/          # Local development (gitignored)
├── dev/            # Development environment
├── staging/        # Staging environment
└── prod/           # Production environment
```

To use a specific environment:
```bash
kedro run --env=prod
```

Or set the environment variable:
```bash
export KEDRO_ENV=prod
kedro run
```

---

## Best Practices

### ✅ DO
- Put all pipeline logic parameters in `parameters.yml`
- Use `local/` for sensitive credentials
- Use environment-specific folders for different deployments
- Keep parameters organized by pipeline/module
- Document parameter meanings with comments

### ❌ DON'T
- Put Prefect orchestration settings here (use `configs/project_config.yaml`)
- Commit sensitive credentials to git
- Hardcode values in node functions (use parameters instead)
- Duplicate parameters across files

---

## Parameter Organization

We organize parameters by pipeline:

```yaml
# Data Processing Pipeline
data_processing:
  download: {...}
  validation: {...}
  cleaning: {...}
  features: {...}

# Sentiment Analysis (part of data_science pipeline)
sentiment_analysis:
  model: {...}
  labels: {...}
  thresholds: {...}

# Topic Modeling (part of data_science pipeline)
topic_modeling:
  model: {...}
  reduction: {...}
  clustering: {...}

# Visualization Pipeline
visualization:
  sentiment_dist: {...}
  topic_dist: {...}
  save: {...}

# Monitoring Pipeline
monitoring:
  sentiment_drift: {...}
  topic_drift: {...}
  alerts: {...}
```

---

## More Information

- [Kedro Configuration Documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html)
- [Kedro Parameters Documentation](https://docs.kedro.org/en/stable/configuration/parameters.html)
- [Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html)
