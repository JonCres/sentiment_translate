# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Core Does

Subscriber Churn Radar: predicts 30/60/90/180/365-day churn probabilities for media & entertainment subscribers using a two-tier modeling approach — a 4-model gradient boosting ensemble (XGBoost, LightGBM, CatBoost, Random Forest) for tabular features, and a CNN-BiLSTM with multi-head self-attention for 90-day temporal sequences. Outputs include risk tiers, hazard rates, SHAP/LIME explanations, and intervention recommendations.

Business context and KPI targets live in `docs/ai_product_canvas.md` and `root_context.md`. All changes must align with those documents.

## Commands

```bash
# Install (from this directory)
uv sync                    # CPU
uv sync --extra mps        # Apple Silicon GPU
uv sync --extra cuda       # NVIDIA GPU

# Run pipelines (sequential: data_processing → data_science → visualization → monitoring)
uv run kedro run --pipeline=data_processing
uv run kedro run --pipeline=data_science
uv run kedro run --pipeline=visualization
uv run kedro run --pipeline=monitoring

# Run full orchestrated pipeline via Prefect
uv run python src/prefect_orchestration/run_all_pipelines.py

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_specific.py::test_function -v

# Linting
uv run black src/
uv run flake8 src/
uv run mypy src/

# Dashboard
uv run streamlit run app/app.py

# MLflow UI (tracking at sqlite:///mlflow.db)
uv run mlflow server --host 127.0.0.1 --port 5000
```

## Architecture

### Four Kedro Pipelines

1. **data_processing** (11 nodes): Raw CSV → skeleton mapping → Pandera validation → RFM feature engineering → 90-day tensor sequences → Feast materialization. Entry point: `src/ai_core/pipelines/data_processing/`.
2. **data_science** (11 nodes): Train CNN-BiLSTM on tensor sequences, train 4 gradient boosters with grid search on tabular features, build soft-voting ensemble, generate multi-horizon predictions, compute SHAP/LIME explanations. Entry point: `src/ai_core/pipelines/data_science/`.
3. **visualization** (3 nodes): SHAP summary plots, model evaluation metrics, model comparison charts. Entry point: `src/ai_core/pipelines/visualization/`.
4. **monitoring** (2 nodes): Data drift detection and model performance monitoring with MLflow logging. Entry point: `src/ai_core/pipelines/monitoring/`.

### The Skeleton Pattern

All raw data passes through "skeleton mapping" before entering pipelines. Each data source (transactions, subscriptions, engagement, QoE, social graph) has a mapping in `conf/base/parameters.yml` under `skeleton:` that renames source columns to canonical names. The mapping logic lives in `src/ai_core/pipelines/data_processing/skeleton.py`. The 7 mandatory skeleton variables are: `customer_id`, `event_timestamp`, `subscription_status`, `transaction_amount`, `session_duration`, `service_error_count`, `account_tenure_days`.

### CNN-BiLSTM Model

Defined in `src/ai_core/pipelines/data_science/dl_nodes.py` as `ChurnCNNAttentionLSTM`:
- Conv1d (local temporal patterns) → BiLSTM (long-range dependencies) → Multi-head attention (event weighting) → Sigmoid output
- Trains on 3D tensor sequences (batch, seq_len=90, features) from `data/05_model_input/tensor_sequences/`
- Returns a `DummyModel` fallback if tensor data is unavailable

### Ensemble Layer

Defined in `src/ai_core/pipelines/data_science/nodes.py`. Four models trained independently with grid search (3-fold CV, ROC-AUC scoring), then combined via `PreFittedVotingClassifier` with soft voting. Prediction node outputs multi-horizon probabilities, hazard rates, risk tiers, and intervention recommendations.

### Prefect Orchestration

`src/core/kedro_pipeline.py` defines the abstract `KedroPipeline` base class that wraps Kedro execution inside Prefect flows. Each pipeline has a corresponding Prefect flow in `src/prefect_orchestration/`. The master flow in `run_all_pipelines.py` runs data → AI sequentially, then visualization and monitoring in parallel.

### Custom Kedro Datasets

- `PolarsDeltaDataset` (`src/ai_core/datasets/polars_delta_dataset.py`): Polars + Delta Lake read/write
- `TensorDataset` (`src/ai_core/datasets/tensor_dataset.py`): NumPy 3D array I/O for DL sequences
- `CloudPickleDataset` (`src/ai_core/datasets/cloudpickle_dataset.py`): Serializes objects with closures/lambdas

### Data Storage

All intermediate and output data uses Delta Lake format (via Polars). Raw inputs are CSV in `data/01_raw/`. Models are tracked in MLflow (experiment: `churn_prediction`, backend: `sqlite:///mlflow.db`). The catalog is defined in `conf/base/catalog.yml` with 26 dataset entries.

## Key Configuration

- **Hyperparameters and skeleton mappings**: `conf/base/parameters.yml` — contains data skeleton column mappings, model hyperparameters, grid search ranges, deep learning config (sequence_length=90, epochs=20, batch_size=64), and explainability settings.
- **Dataset definitions**: `conf/base/catalog.yml` — maps Kedro dataset names to file paths and types.
- **Prefect deployment config**: `configs/project_config.yaml` — flow entrypoints, schedules, retry/timeout settings, work pool config.
- **Local overrides** (gitignored): `conf/local/credentials.yml`, `conf/local/mlflow.yml`.

## Validation

- **DataFrame schemas**: `src/ai_core/schemas.py` uses Pandera (Polars backend) for `TransactionSchema`, `RFMSchema`, `SubscriptionSchema`, `QoESchema`, `InteractionSchema`.
- **Config validation**: Pydantic models `SkeletonParams` and `FeastConfig` in the same file.
- Validate within Kedro nodes, not at pipeline boundaries.

## Core-Specific Rules

- **Temporal splitting**: Train on T-365 to T-90, validate T-90 to T-30, test T-30 to T-0. Never leak future data.
- **SMOTE**: Apply only to training set, never to validation or test.
- **Churn label**: `inactive_duration > 90 days` (configurable via `inactivity_threshold_days` in parameters).
- **PII**: Subscriber IDs must be salted SHA-256 hashed. IPs truncated via /24.
- **Logging**: Use `logging.getLogger(__name__)`, never `print()`.
- **Hardware detection**: `src/utils/device.py` auto-detects CUDA/MPS/CPU. The CNN-BiLSTM trains on GPU when available.
