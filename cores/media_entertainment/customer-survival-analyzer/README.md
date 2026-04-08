# Media Entertainment: Customer Survival Analyzer 🚀

## Table of Contents

- [Media Entertainment: Customer Survival Analyzer 🚀](#media-entertainment-customer-survival-analyzer-)
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
  - [1. Ideal Data Profile: **User Interaction \& Subscription Logs**](#1-ideal-data-profile-user-interaction--subscription-logs)
  - [2. Mandatory Variables \& Schema](#2-mandatory-variables--schema)
  - [3. Recommended Benchmark Datasets](#3-recommended-benchmark-datasets)
    - [A. **KKBox Music Streaming Churn Data (Kaggle)**](#a-kkbox-music-streaming-churn-data-kaggle)
    - [B. **Sparkify Digital Music Service (Udacity/GitHub)**](#b-sparkify-digital-music-service-udacitygithub)
    - [C. **Telco Customer Churn (Kaggle/IBM)**](#c-telco-customer-churn-kaggleibm)
  - [Survival Analysis Framework](#survival-analysis-framework)
  - [Summary for your Project](#summary-for-your-project)
  - [Hybrid Architecture: Prefect + Kedro](#hybrid-architecture-prefect--kedro)
    - [Quick Overview](#quick-overview)
    - [Key Features](#key-features)
    - [Documentation](#documentation)

## Technical recommendations before starting

This project uses **[uv](https://docs.astral.sh/uv/)** as its primary package and environment manager for superior performance and dependency resolution. All commands in this README assume you're using `uv` properly.

### Core Principles

** ✅ ALWAYS use `uv run` to execute commands** (kedro, streamlit, mlflow, pytest, python scripts, etc.)

** ❌ AVOID activating the virtual environment manually** unless you have a specific reason

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
uv python install 3.12
uv python pin 3.12
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

## 1. Ideal Data Profile: **User Interaction & Subscription Logs**

The ideal dataset for this AI Core involves a **user interaction log** where **customers** exhibit **engagement decay and hazard factors** toward specific **streaming services**.

**Why this fits:**

- **Granularity:** Data should capture **temporal sequences and frustration indices** to enable high-accuracy **continuous time-to-event modeling**.
- **Temporal Depth:** Historical logs should span at least **12-24 months** to establish baseline tenure and hazard rates.

---

## 2. Mandatory Variables & Schema

To successfully execute the core pipelines, your dataset should contain variables covering Transactions (Non-Contractual), Subscriptions (Contractual), and Engagement (Deep Learning):

| Variable Name         | Description                                        | Data Type      |
| :-------------------- | :------------------------------------------------- | :------------- |
| **`customer_id`**     | Unique user identifier.                            | `str`          |
| **`subscription_id`** | Contract identifier.                               | `str`          |
| **`start_date`**      | Acquisition or contract start date.                | `datetime`     |
| **`end_date`**        | Churn date (null if active).                       | `datetime`     |
| **`transaction_date`**| Date of purchase/interaction.                      | `datetime`     |
| **`volume`**          | Consumption volume (e.g., minutes).                | `int`          |
| **`sequences`**       | 3D Temporal User Embeddings for deep learning.     | `tensor`       |

---

## 3. Recommended Benchmark Datasets

The following public datasets are recommended for establishing a baseline for this AI Core:

### A. **KKBox Music Streaming Churn Data (Kaggle)**

- **Description:** Leading music streaming service dataset with granular usage logs.
- **Relevance:** The gold standard for fine-tuning continuous hazard rate models.
- **Source:** [KKBox Dataset on Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)

### B. **Sparkify Digital Music Service (Udacity/GitHub)**

- **Description:** Synthetic dataset simulating event-level interactions.
- **Relevance:** Ideal for deep behavioral modeling and Frustration Index calculation.
- **Source:** [Sparkify Project Source](https://github.com/yduh/Churn-Prediction)

### C. **Telco Customer Churn (Kaggle/IBM)**

- **Description:** Classic dataset for survival analysis benchmarking.
- **Relevance:** Excellent for comparing CPH and KM curves across tiers.
- **Source:** [Telco Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Survival Analysis Framework

Attrition is modeled as the instantaneous hazard rate $h(t)$, shifting the focus from binary churn to **when** a user will disengage:

1. **Statistical Hazard Modeling:** CoxPH and Kaplan-Meier curves for baseline benchmarking.
2. **Machine Learning Survival:** Random Survival Forests (RSF) for non-linear interactions in telemetry.
3. **Deep Survival Track:** DeepSurv and N-MTLR optimized for continuous time-to-event modeling.

## Summary for your Project

1. **Ingest Raw Data:** Load source data into the `01_raw` (Bronze) layer.
2. **Transform & Engineer:** Run the `data_processing` pipeline to generate the `04_feature` layer.
3. **Train & Predict:** Execute the `data_science` pipeline to generate models and predictions.
4. **Iterate & Visualize:** Use the Streamlit dashboard (`app/`) to evaluate performance and business impact.

---

## Hybrid Architecture: Prefect + Kedro

This AI Core uses a **hybrid architecture** that combines:

- **Prefect 2.x/3.x**: Orchestration, scheduling, observability, and deployment
- **Kedro 0.19+**: Data pipeline definition, reproducibility, and visualization
- **Feast 0.38+**: Feature store for versioned feature engineering
- **MLflow**: Model registry and experiment tracking

### Quick Overview

The architecture separates concerns into distinct layers:

1. **Presentation Layer** (`app/`): Streamlit dashboards for business users
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

<-backend-config=profile=AdministratorAccess-716773066006 AICR-280 version seed -->
<-backend-config=profile=AdministratorAccess-716773066006 AICR-280 major bump test -->
<-backend-config=profile=AdministratorAccess-716773066006 AICR-280 cleanup + major tag test -->
