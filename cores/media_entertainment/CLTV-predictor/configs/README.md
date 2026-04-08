# Prefect Orchestration Configuration Directory

## 📑 Table of Contents

- [Configuration Directory](#configuration-directory)
  - [📑 Table of Contents](#-table-of-contents)
  - [Files](#files)
    - [`project_config.yaml`](#project_configyaml)
  - [Kedro Pipeline Parameters](#kedro-pipeline-parameters)
  - [Configuration Separation](#configuration-separation)
  - [Example Usage](#example-usage)
    - [In Prefect Flows](#in-prefect-flows)
    - [In Kedro Nodes](#in-kedro-nodes)
    - [In Kedro Pipeline Definitions](#in-kedro-pipeline-definitions)
  - [Environment-Specific Configuration](#environment-specific-configuration)
  - [Migration Notes](#migration-notes)

This directory contains **Prefect orchestration configuration** following standard conventions for the **CLTV predictor** project.

## Files

### `project_config.yaml`

**Purpose**: Prefect orchestration settings

**Contains**:

- Project metadata (name, version, industry)
- Prefect server configuration (API URL, work pool settings)
- Deployment configurations (names, tags, schedules, entrypoints)
- Kedro integration settings (which pipelines to orchestrate)
- Alerting and notification settings
- Execution settings (retries, timeouts, concurrency)
- Logging configuration for Prefect

**Used by**:

- `src/core/base_pipeline.py` - Loads this config for all Prefect pipeline wrappers
- `src/prefect_orchestration/*.py` - All Prefect flow definitions

**DO NOT** put pipeline logic parameters here (data processing settings, model hyperparameters, etc.)

---

## Kedro Pipeline Parameters

For **pipeline logic parameters** (data sources, validation rules, model settings, etc.), use:

📁 **`conf/base/parameters.yml`** ← Kedro configuration

This is the standard Kedro location for pipeline parameters and is automatically loaded by Kedro when running pipelines.

---

## Configuration Separation

We maintain a clear separation:

| Configuration Type | File                          | Purpose                                       | Loaded By       |
| ------------------ | ----------------------------- | --------------------------------------------- | --------------- |
| **Orchestration**  | `configs/project_config.yaml` | Prefect deployment, scheduling, work pools    | Prefect flows   |
| **Pipeline Logic** | `conf/base/parameters.yml`    | Data processing, models, features, thresholds | Kedro pipelines |

This separation ensures:

- ✅ No redundancy
- ✅ Clear separation of concerns
- ✅ Follows Kedro best practices
- ✅ Easy to maintain and understand

---

## Example Usage

### In Prefect Flows

```python
from src.core.base_pipeline import BasePipeline

class DataPipeline(BasePipeline):
    def __init__(self):
        # Loads configs/project_config.yaml
        super().__init__("configs/project_config.yaml")
    
    def run(self):
        # Access Prefect settings
        work_pool = self.config['prefect']['work_pool']['name']
        
        # Run Kedro pipeline (which uses conf/base/parameters.yml)
        session.run(pipeline_name="data_processing")
```

### In Kedro Nodes

```python
def download_data(params: Dict[str, Any]) -> pd.DataFrame:
    # params comes from conf/base/parameters.yml
    dataset = params['dataset']
    sample_size = params['sample_size']
    # ...
```

### In Kedro Pipeline Definitions

```python
node(
    func=download_data,
    inputs="params:data_processing.download",  # ← From conf/base/parameters.yml
    outputs="raw_data",
)
```

---

## Environment-Specific Configuration

For environment-specific overrides:

- **Prefect**: Modify `configs/project_config.yaml` or use environment variables
- **Kedro**: Use `conf/local/parameters.yml` (gitignored) or environment-specific folders

---

## Migration Notes

If you're migrating from the old structure where `configs/config.yaml` contained both Prefect and Kedro settings:

1. ✅ Prefect settings → `configs/project_config.yaml`
2. ✅ Kedro pipeline parameters → `conf/base/parameters.yml`
3. ✅ All pipeline files updated to use `project_config.yaml`
4. ✅ No redundancy between files
