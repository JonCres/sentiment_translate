# Voice of Customer: Configuration Guide 🚀

## Table of Contents

- [Voice of Customer: Configuration Guide 🚀](#voice-of-customer-configuration-guide-)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
    - [`project_config.yaml`](#project_configyaml)
  - [Kedro Pipeline Parameters](#kedro-pipeline-parameters)
  - [Configuration Separation](#configuration-separation)
  - [Configuration Files](#configuration-files)
    - [`project_config.yaml`](#project_configyaml-1)
    - [`conf/base/parameters.yml`](#confbaseparametersyml)
    - [`conf/base/catalog.yml`](#confbasecatalogyml)
    - [`conf/local/credentials.yml`](#conflocalcredentialsyml)
  - [Example Usage](#example-usage)
    - [In Prefect Flows](#in-prefect-flows)
    - [In Kedro Nodes](#in-kedro-nodes)
  - [Hybrid Architecture: Prefect + Kedro](#hybrid-architecture-prefect--kedro)

This directory contains **Prefect orchestration configuration** only.

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

We maintain a clear separation of concerns to ensure maintainability and security:

| Configuration Type | File                          | Purpose                                       | Loaded By       |
| ------------------ | ----------------------------- | --------------------------------------------- | --------------- |
| **Orchestration**  | `configs/project_config.yaml` | Prefect deployment, scheduling, work pools    | Prefect flows   |
| **Pipeline Logic** | `conf/base/parameters.yml`    | Data processing, models, features, thresholds | Kedro pipelines |

---

## Configuration Files

### `project_config.yaml`
**Purpose**: Prefect orchestration settings.
**Contains**: Project metadata, Prefect server config, deployment schedules, and execution settings (retries, timeouts).

### `conf/base/parameters.yml`
**Purpose**: Kedro pipeline logic.
**Contains**: Data sources, validation rules, model hyperparameters, and pipeline-specific thresholds.

### `conf/base/catalog.yml`
**Purpose**: Data abstraction layer.
**Contains**: Dataset definitions and storage locations (S3, Delta Lake, local).

### `conf/local/credentials.yml`
**Purpose**: Secret management (GITIGNORED).
**Contains**: API keys, database passwords, and cloud credentials.

---

## Example Usage

### In Prefect Flows
```python
from src.core.base_pipeline import KedroPipeline

class DataPipeline(KedroPipeline):
    def __init__(self):
        # Loads configs/project_config.yaml
        super().__init__("configs/project_config.yaml")
```

### In Kedro Nodes
```python
def download_data(params: Dict[str, Any]) -> pd.DataFrame:
    # params comes from conf/base/parameters.yml
    dataset = params['dataset']
```

---

## Hybrid Architecture: Prefect + Kedro

This AI Core utilizes a hybrid architecture that separates orchestration from transformation.

- **Prefect**: Manages the "When" and "How" of execution (scheduling, retries, observability).
- **Kedro**: Manages the "What" and "Where" of data transformation (pipelines, data catalog).
- **Feast**: Manages the "Which" of feature versioning.
- **MLflow**: Manages the "Who" of model lineage.

**See:** [`docs/technical_design.md`](../docs/technical_design.md) - Section 2.6: Hybrid Architecture Implementation Details
