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

This directory contains **Kedro pipeline configuration** following standard Kedro conventions for the **Customer survival analyzer** project.

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
- **AI Modeling**: Model types (survival_analysis) and hyperparameters for CoxPH, Random Survival Forest, and Deep Learning (DeepSurv).
- **Visualization**: Plot themes, directory paths, and survival curve settings.
- **MLOps**: Experiment tracking (MLflow), metric definitions (C-Index), and drift detection thresholds.

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
  model_type: "survival_analysis"
  survival_analysis:
    cph:
      penalizer: 0.1
    rsf:
      n_estimators: 100

visualization:
  plot_theme: "seaborn-darkgrid"
  survival_curves:
    time_horizon_days: 90

mlops:
  experiment_name: "survival_analysis"
  metrics: ["c_index", "brier_score", "faithfulness_score"]
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
    func=train_coxph_model,
    inputs=["survival_data_prepared", "parameters"],  # ← Reads from parameters.yml
    outputs="cox_ph_model",
)
```

### 2. Parameters in Node Functions

```python
# In src/ai_core/pipelines/data_science/nodes.py
def train_coxph_model(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> CoxPHFitter:
    cph_params = parameters.get("modeling", {}).get("survival_analysis", {}).get("cph", {})
    cph = CoxPHFitter(penalizer=cph_params.get("penalizer", 0.1))
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
  model_type: "survival_analysis"
  survival_analysis:
    cph: {...}
    rsf: {...}
    deepsurv: {...}

# 4. Visualization (Plot settings)
visualization:
  plots_dir: "..."
  plot_theme: "..."
  survival_curves: {...}

# 5. Monitoring (MLOps Tracking)
mlops:
  experiment_name: "..."
  drift_detection: {...}
```

---

## More Information

- [Kedro Configuration Documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html)
- [Kedro Parameters Documentation](https://docs.kedro.org/en/stable/configuration/parameters.html)
- [Kedro Data Catalog](https://docs.kedro.org/en/stable/data/data_catalog.html)
