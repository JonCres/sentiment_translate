# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Predictive CLTV Insights** (a.k.a. Aura Value Predictor) — a Media & Entertainment AI Core that predicts Customer Lifetime Value using a hybrid probabilistic + ML approach. Targets SVOD, TVOD, and F2P Gaming domains.

**Python 3.13** (>=3.13, <3.14). Managed with **uv** (not pip). Kedro package name: `ai_core`.

Business context and KPI targets live in `root_context.md` and `docs/ai_canvas.md`. All changes must align with these documents.

## Commands

All commands run from this directory (`cores/media_entertainment/predictive-CLTV-insights/`).

```bash
# Install dependencies
uv sync
uv sync --extra xpu     # Intel Arc GPU (Windows only)
uv sync --extra cuda    # NVIDIA GPU
uv sync --extra mps     # Apple Silicon

# Run pipelines
uv run kedro run --pipeline=data_processing
uv run kedro run --pipeline=data_science
uv run kedro run --pipeline=visualization
uv run kedro run --pipeline=monitoring

# Run specific node
uv run kedro run --pipeline=data_science --node=train_bg_nbd_model

# Visualize pipeline DAG
uv run kedro viz

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_xpu.py::test_function -v

# Code quality
black src/
flake8 src/
mypy src/

# Streamlit dashboard
uv run streamlit run app/app.py

# MLflow tracking
uv run mlflow server --host 127.0.0.1 --port 5000

# Prefect orchestration
uv run prefect server start
uv run python src/prefect_orchestration/run_all_pipelines.py
```

## Architecture

### Three-Layer Pattern

1. **Orchestration** (`src/prefect_orchestration/`): Prefect `@flow`/`@task` decorators for scheduling, retries, and monitoring. `run_all_pipelines.py` is the master flow that runs data → AI sequentially, then visualization + monitoring in parallel.

2. **Transformation** (`src/ai_core/pipelines/`): Kedro pipelines with pure-function nodes. Four pipelines:
   - `data_processing` — Raw data → skeleton mapping → RFM features → Feast materialization
   - `data_science` — BG/NBD, Gamma-Gamma, sBG, Weibull AFT training + XGBoost residual refinement
   - `visualization` — Lifetimes plots + SLM-generated interpretations (Phi-3.5-mini)
   - `monitoring` — Distribution drift detection with threshold alerting

3. **Presentation** (`app/`): Streamlit dashboard consuming Delta Lake tables and model artifacts.

### Kedro-Prefect Adapter

`src/core/kedro_pipeline.py` defines the `KedroPipeline` abstract base class that bridges Kedro sessions into Prefect tasks. Each pipeline subclass (e.g., `DataPipeline`, `AIPipeline`) sets `pipeline_name` and gets auto-wrapped as a `@flow`.

### Data Flow

Raw CSV → **Skeleton Pattern** (source-agnostic mapping via `skeleton.py`) → Delta Lake intermediates → RFM/Survival features → Model training → CLTV predictions → Reporting plots + SLM narratives.

All datasets defined in `conf/base/catalog.yml`. Custom dataset types:
- `PolarsDeltaDataset` (`src/ai_core/datasets/`) — Polars DataFrames backed by Delta Lake
- `CloudPickleDataset` — Model serialization
- `MlflowModelTrackingDataset` — MLflow registry integration

### Models

**Non-Contractual (BTYD):** BG/NBD (purchase frequency) + Gamma-Gamma (monetary value), with optional XGBoost residual refinement (`CLTV_Final = CLTV_Statistical + Δ_XGBoost`).

**Contractual (Survival):** sBG (discrete churn) + Weibull AFT (continuous churn with engagement covariates). Validated via Kaplan-Meier and Cox PH analysis.

**XGBoost Refinement Features:** QoE (buffering ratios), Breadth (catalog exploration), Velocity (usage intensity changes).

### Hardware Acceleration

`src/utils/device.py` provides `get_device()` which detects: Intel Arc XPU → CUDA → MPS → CPU. XGBoost supports `device='sycl'` for Intel Arc. PyTorch models (SLM inference) use the detected device automatically.

### Feature Store

Feast configuration in `feature_repo/`. Three feature views: `rfm_features`, `survival_features`, `behavioral_features`. Backed by Delta Lake with 90-day TTL. Features are materialized at the end of the `data_processing` pipeline.

## Critical Rules

1. **The Pickling Rule**: Never pass `KedroContext`, `KedroSession`, or `DataCatalog` between Prefect tasks. Pass config key strings and re-initialize `KedroSession` inside each task.
2. **Reproducibility**: All random seeds must be `42` (`random_state=42`).
3. **Data Purity**: Exclude first purchase from monetary calculations to avoid CAC distortion.
4. **No hardcoded paths**: Always use the Kedro DataCatalog for I/O.
5. **Validation**: Use Pandera schemas (`src/ai_core/schemas.py`) for DataFrame validation in nodes.
6. **Type hints**: Mandatory. Use Pydantic for config/API validation.
7. **Docstrings**: Google Style with Args, Returns, Raises.
8. **Formatting**: Black (line length 100), Flake8 (ignore E203, W503), mypy strict.

## Configuration

- `conf/base/catalog.yml` — Dataset definitions (Delta Lake, MLflow, CloudPickle)
- `conf/base/parameters.yml` — Skeleton mappings, model hyperparameters, SLM config, MLOps settings
- `conf/local/` — Gitignored: `credentials.yml`, `mlflow.yml`
- `configs/project_config.yaml` — Prefect deployment: work pools, schedules, retries, timeouts

## Key Patterns

**Adding a new node:**
```python
# 1. Write the function in the appropriate nodes.py
def my_node(data: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
    """Short description.

    Args:
        data: Input dataframe.
        params: Parameters from conf/base/parameters.yml.

    Returns:
        Processed dataframe.
    """
    return processed_data

# 2. Register in the corresponding pipeline.py
node(func=my_node, inputs=["input_dataset", "params:section"], outputs="output_dataset", name="my_node_name")

# 3. Define dataset in conf/base/catalog.yml
# output_dataset:
#   type: ai_core.datasets.polars_delta_dataset.PolarsDeltaDataset
#   filepath: data/07_model_output/my_output
#   save_args:
#     mode: overwrite
```