# Media Entertainment: Churn Forecasting 🚀

## Table of Contents

- [Media Entertainment: Churn Forecasting 🚀](#media-entertainment-churn-forecasting-)
  - [Table of Contents](#table-of-contents)
  - [Technical recommendations before starting](#technical-recommendations-before-starting)
    - [Core Principles](#core-principles)
    - [Why UV?](#why-uv)
    - [Command Execution Patterns](#command-execution-patterns)
      - [✅ Recommended: Direct UV Execution](#-recommended-direct-uv-execution)
      - [⚠️ Alternative: Manual Environment Activation](#️-alternative-manual-environment-activation)
      - [❌ Anti-Pattern: Direct Commands Without UV](#-anti-pattern-direct-commands-without-uv)
    - [Dependency Installation Best Practices](#dependency-installation-best-practices)
      - [Installing Project Dependencies](#installing-project-dependencies)
      - [Understanding UV's Dependency Management](#understanding-uvs-dependency-management)
    - [Quick Reference Table](#quick-reference-table)
    - [Common Issues and Solutions](#common-issues-and-solutions)
    - [Best Practices Summary](#best-practices-summary)
  - [1. Ideal Data Profile: **Subscription Logs \& User Interaction Events**](#1-ideal-data-profile-subscription-logs--user-interaction-events)
  - [2. Mandatory Variables \& Schema](#2-mandatory-variables--schema)
  - [3. Recommended Benchmark Datasets](#3-recommended-benchmark-datasets)
    - [A. **KKBox Music Streaming Churn Data (Kaggle)**](#a-kkbox-music-streaming-churn-data-kaggle)
    - [B. **Sparkify Digital Music Service (Udacity/GitHub)**](#b-sparkify-digital-music-service-udacitygithub)
    - [C. **Telco Customer Churn (Kaggle/IBM)**](#c-telco-customer-churn-kaggleibm)
  - [How Churn is Predicted](#how-churn-is-predicted)
  - [Summary for your Project](#summary-for-your-project)
  - [Hybrid Architecture: Prefect + Kedro](#hybrid-architecture-prefect--kedro)
    - [Quick Overview](#quick-overview)
    - [Key Features](#key-features)
    - [Documentation](#documentation)

## Technical recommendations before starting

This project uses **[uv](https://docs.astral.sh/uv/)** as its primary package and environment manager for superior performance and dependency resolution. All commands in this README assume you're using `uv` properly.

### Core Principles

**✅ ALWAYS use `uv run` to execute commands** (kedro, streamlit, mlflow, pytest, python scripts, etc.)

**❌ AVOID activating the virtual environment manually** unless you have a specific reason

### Why UV?

- **10-100x faster** than pip for dependency resolution and installation
- **Consistent environments** across all team members
- **Built-in project management** without needing separate virtualenv tools
- **Automatic environment activation** via `uv run`

---

### Command Execution Patterns

#### ✅ Recommended: Direct UV Execution

```bash
# Run Kedro pipelines
uv run kedro run --pipeline=data_processing
uv run kedro viz

# Run Streamlit apps
uv run streamlit run app/main.py

# Start MLflow server
uv run mlflow server --host 127.0.0.1 --port 5000

# Run tests
uv run pytest tests/unit/ -v

# Execute Python scripts
uv run python scripts/check_metrics.py --pipeline customer-churn --days 7

# Run Prefect commands
uv run prefect server start
uv run prefect worker start -p "aicore-pool"
```

#### ⚠️ Alternative: Manual Environment Activation

Only use this if you need to run multiple commands in succession:

```bash
# Activate the virtual environment (created by uv sync)
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Now you can run commands without 'uv run' prefix
kedro run --pipeline=data_processing
streamlit run app/main.py
pytest tests/

# Remember to deactivate when done
deactivate
```

#### ❌ Anti-Pattern: Direct Commands Without UV

```bash
# DON'T DO THIS - bypasses uv environment management
kedro run                    # ❌ May use wrong Python/packages
streamlit run app/main.py    # ❌ Dependencies might not match
pytest tests/                # ❌ Unpredictable behavior
```

---

### Dependency Installation Best Practices

#### Installing Project Dependencies

```bash
# ✅ RECOMMENDED: Sync all dependencies (respects pyproject.toml and uv.lock)
uv sync

# ✅ RECOMMENDED: Install with hardware acceleration
uv sync --extra mps   # Apple Silicon (M1/M2/M3)
uv sync --extra cuda  # NVIDIA GPUs
uv sync --extra xpu   # Intel Arc GPUs

# ✅ Install additional packages (automatically updates pyproject.toml)
uv add package-name
uv add --dev package-name  # Development dependency

# ❌ AVOID: Using pip directly
pip install -r requirements.txt  # Bypasses uv's lock file and benefits
```

#### Understanding UV's Dependency Management

- **`uv sync`**: Installs/syncs dependencies from `pyproject.toml` and `uv.lock`
- **`uv add package`**: Adds package to `pyproject.toml` and installs it
- **`uv pip install`**: Lower-level command, prefer `uv sync` or `uv add`
- **`uv run`**: Automatically ensures environment is synced before running

---

### Quick Reference Table

| Task | ✅ Correct Command | ❌ Incorrect Command |
|------|-------------------|---------------------|
| Install dependencies | `uv sync` | `pip install -r requirements.txt`|
| Add new package | `uv add pandas` | `pip install pandas` |
| Run Kedro pipeline | `uv run kedro run` | `kedro run` |
| Start Streamlit | `uv run streamlit run app/main.py` | `streamlit run app/main.py` |
| Run tests | `uv run pytest tests/` | `pytest tests/` |
| Start MLflow | `uv run mlflow server` | `mlflow server` |
| Execute Python script | `uv run python script.py` | `python script.py` |

---

### Common Issues and Solutions

**Issue**: "Command not found" error when running `kedro`, `streamlit`, etc.

**Solution**: Always prefix with `uv run`, or activate the environment first with `source .venv/bin/activate`

---

**Issue**: Dependencies seem outdated or mismatched

**Solution**: Re-sync the environment:

```bash
uv sync --refresh
```

---

**Issue**: Need to use a different Python version

**Solution**: UV can manage Python versions:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
```

---

### Best Practices Summary

1. **Always use `uv run`** for executing commands
2. **Use `uv sync`** to install/update dependencies
3. **Use `uv add`** to add new packages
4. **Commit `uv.lock`** to version control for reproducible builds
5. **Never use bare `pip install`** - it bypasses uv's dependency resolution

---

## 1. Ideal Data Profile: **Subscription Logs & User Interaction Events**

The ideal dataset for this AI Core involves **subscription and interaction logs** where **customers** exhibit **churn indicators** toward specific **SVOD/AVOD services**.

**Why this fits:**

- **Granularity:** Data should capture the **temporal sequence of events** (Logins, Watch Time, QoE) to enable high-accuracy **ensemble and deep learning modeling**.
- **Temporal Depth:** Historical logs should span at least **6-12 months** to account for seasonality and trend shifts.

---

## 2. Mandatory Variables & Schema

To successfully execute the core pipelines, your input dataset should contain the following variables:

| Variable Name         | Description                                        | Data Type      |
| :-------------------- | :------------------------------------------------- | :------------- |
| **`customer_id`**     | Unique identifier for the primary atomic unit.     | `str`          |
| **`start_date`**      | When the subscription started.                     | `datetime`     |
| **`end_date`**        | When the subscription ended (null if active).      | `datetime`     |
| **`watch_time`**      | Consumption volume in minutes.                     | `float`        |
| **`session_count`**   | Number of active sessions.                         | `int`          |
| **`buffering_events`**| Number of technical issues (QoE).                  | `int`          |

---

## 3. Recommended Benchmark Datasets

The following public datasets are recommended for establishing a baseline for this AI Core:

### A. **KKBox Music Streaming Churn Data (Kaggle)**

- **Description:** Leading music streaming service dataset with transactional history and usage logs.
- **Relevance:** Highly suitable for modeling behavioral decay.
- **Source:** [KKBox Dataset on Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)

### B. **Sparkify Digital Music Service (Udacity/GitHub)**

- **Description:** Synthetic dataset simulating a service like Spotify (26M+ rows).
- **Relevance:** Ideal for deep behavioral modeling and sequential patterns.
- **Source:** [Sparkify Project Source](https://github.com/yduh/Churn-Prediction)

### C. **Telco Customer Churn (Kaggle/IBM)**

- **Description:** Classic dataset for churn benchmarking with service-level attributes.
- **Relevance:** Excellent for tabular ensemble baseline models.
- **Source:** [Telco Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## How Churn is Predicted

Churn forecasting is handled via a multi-layered approach:

1. **Ensemble Layer:** Combining XGBoost, LightGBM, and CatBoost for robust risk scoring on tabular aggregates.
2. **Deep Learning Layer:** CNN-BiLSTM architecture modeling temporal viewing sequences to detect subtle decay.
3. **Business Logic Layer:** Multi-horizon probability calculations with priority assignment for targeted interventions.

## Summary for your Project

1. **Load Data:** Ensure your DataFrame has `customer_id`, `interaction_timestamp`, and engagement events (watch time, session duration, QoE).
2. **Transform:** Engineer sequential behavior and tabular snapshots.
3. **Fit CNN-BiLSTM:** Model temporal viewing patterns for deep behavioral insight.
4. **Fit Ensemble:** Soft-vote across XGBoost, LightGBM, and CatBoost for robust risk scoring.
5. **Predict Survival:** Calculate multi-horizon churn probabilities, hazard rates, and predicted tenure.

---

## Hybrid Architecture: Prefect + Kedro

This AI Core uses a **hybrid architecture** that combines:

- **Prefect 2.x/3.x**: Orchestration, scheduling, observability, and deployment
- **Kedro 0.19+**: Data pipeline definition, reproducibility, and visualization
- **Feast 0.38+**: Feature store for versioned feature engineering
- **MLflow**: Model registry and experiment tracking

### Quick Overview

The architecture separates concerns into distinct layers:

1. **Presentation Layer** (`app/`): Streamlit dashboards for executive decision-making
2. **Orchestration Layer** (`src/prefect_orchestration/`): Prefect flows that schedule and monitor workflows
3. **Transformation Layer** (`src/aicore/`): Kedro pipelines for data processing, model training, and visualization
4. **Data & Storage Layer**: Feast feature store, MLflow model registry, and data catalog

### Key Features

- **Robust Workflow Management:** Prefect ensures reliable execution, retries, and monitoring of all pipeline runs.
- **Reproducible Data Pipelines:** Kedro provides a modular and version-controlled framework for data transformations and model training.
- **Scalable Feature Management:** Feast efficiently stores and serves engineered features for both training and inference.
- **Comprehensive Observability:** Integration with MLflow for experiment tracking and model registry, enhancing model lifecycle management.

### Documentation

For comprehensive implementation details, including:
- Complete project structure
- Standardized dataset usage (PolarsDeltaDataset, CloudPickleDataset)
- Detailed workflow diagrams
- Configuration patterns
- Feast feature store setup and usage
- Extension guidelines

**See:** [`docs/technical_design.md`](docs/technical_design.md) - Section 2.6: Hybrid Architecture Implementation Details

---
