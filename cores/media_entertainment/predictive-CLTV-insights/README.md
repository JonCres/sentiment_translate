# Media Entertainment: Predictive CLTV Insights 🚀

## Table of Contents

- [Media Entertainment: Predictive CLTV Insights 🚀](#media-entertainment-predictive-cltv-insights-)
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
  - [1. Ideal Data Profile: **In-Game Microtransactions or Digital Content Purchases**](#1-ideal-data-profile-in-game-microtransactions-or-digital-content-purchases)
  - [2. Mandatory Variables \& Schema](#2-mandatory-variables--schema)
  - [3. Recommended Benchmark Datasets](#3-recommended-benchmark-datasets)
    - [A. **KKBox Music Streaming Churn Data (Kaggle)**](#a-kkbox-music-streaming-churn-data-kaggle)
    - [B. **Sparkify Digital Music Service (Udacity/GitHub)**](#b-sparkify-digital-music-service-udacitygithub)
    - [C. **Telco Customer Churn (Kaggle/IBM)**](#c-telco-customer-churn-kaggleibm)
  - [How CLTV is Computed](#how-cltv-is-computed)
    - [1. Non-Contractual Models (Gaming/TVOD)](#1-non-contractual-models-gamingtvod)
    - [2. Contractual Models (SVOD/Subscriptions)](#2-contractual-models-svodsubscriptions)
    - [3. ML Refinement (XGBoost)](#3-ml-refinement-xgboost)
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
uv run streamlit run app/app.py

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
uv sync --extra xpu   # Intel Arc GPUs (Windows only - see Hardware Acceleration section below)

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

### Hardware Acceleration Support

This AI Core supports multiple hardware acceleration backends for faster model training and inference:

#### Supported Hardware

| Platform | Hardware | Status | Command |
|----------|----------|--------|---------|
| 🪟 Windows | Intel Arc GPU (XPU) | ✅ Full Support | `uv sync --extra xpu` |
| 🪟 Windows | NVIDIA GPU (CUDA) | ✅ Full Support | `uv sync --extra cuda` |
| 🐧 Linux | NVIDIA GPU (CUDA) | ✅ Full Support | `uv sync --extra cuda` |
| 🐧 Linux | Intel Arc GPU (XPU) | ⚠️ Manual Setup Required | See below |
| 🍎 macOS | Apple Silicon (MPS) | ✅ Full Support | `uv sync --extra mps` |
| All | CPU Only | ✅ Full Support | `uv sync` |

#### Intel Arc GPU (XPU) Setup

**Windows (Automated):**

```bash
# 1. Install Intel Arc GPU drivers from Intel's website
# 2. Install PyTorch XPU
uv sync --extra xpu

# 3. Verify installation
uv run python test_xpu_setup.py
```

**Linux (Manual Setup Required):**

Due to PyTorch packaging limitations, Linux XPU requires manual installation:

```bash
# 1. Install Intel compute runtime (Level Zero drivers)
sudo apt install intel-level-zero-gpu intel-opencl-icd

# 2. Install base dependencies
uv sync

# 3. Manually install PyTorch XPU (outside uv environment)
# Note: This step is required due to missing pytorch-triton-xpu packages in PyPI
.venv/bin/pip install torch==2.6.0+xpu torchvision==0.21.0+xpu torchaudio==2.6.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu

# 4. Verify XPU is available
uv run python test_xpu_setup.py
```

**Why is Linux XPU manual?**

PyTorch's XPU index for Linux is missing the `pytorch-triton-xpu` dependency packages that are required by newer PyTorch versions. This is a known limitation as of 2025. Intel Extension for PyTorch (IPEX) has reached EOL, and PyTorch now handles XPU natively on Windows, but Linux support requires workarounds until PyTorch resolves the packaging issue.

#### XGBoost Hardware Acceleration

XGBoost automatically uses the detected hardware backend:

- **CUDA**: GPU-accelerated training with `device='cuda'`
- **XPU**: GPU-accelerated training with `device='sycl'` (Intel oneAPI)
- **CPU**: Falls back to CPU with optional Intel oneDAL optimization

The `src/utils/device.py` module automatically detects and configures the optimal backend.

---

### Best Practices Summary

1. **Always use `uv run`** for executing commands
2. **Use `uv sync`** to install/update dependencies
3. **Use `uv add`** to add new packages
4. **Commit `uv.lock`** to version control for reproducible builds
5. **Never use bare `pip install`** - it bypasses uv's dependency resolution

---

## 1. Ideal Data Profile: **In-Game Microtransactions or Digital Content Purchases**

The ideal dataset for this AI Core involves **transactional logs** where **customers** exhibit **purchasing behavior** toward specific **non-contractual goods**.

**Why this fits:**

- **Granularity:** Data should capture **irregular frequency and varying monetary value** to enable high-accuracy **BTYD and XGBoost modeling**.
- **Temporal Depth:** Historical logs should span at least **12 months** to account for seasonality and trend shifts.

---

## 2. Mandatory Variables & Schema

To successfully execute the core pipelines, your input dataset should contain the following variables:

| Variable Name           | Description                                                        | Data Type              |
| :---------------------- | :----------------------------------------------------------------- | :--------------------- |
| **`customer_id`**       | Unique user identifier.                                            | `str` / `int`          |
| **`transaction_date`**  | Date of purchase, used for Recency ($t_x$) and Tenure ($T$).       | `datetime`             |
| **`transaction_value`** | Monetary amount of the purchase.                                  | `float`                |

---

## 3. Recommended Benchmark Datasets

The following public datasets are recommended for establishing a baseline for this AI Core:

### A. **KKBox Music Streaming Churn Data (Kaggle)**

- **Description:** Leading music streaming service dataset with transactional history and usage logs.
- **Relevance:** Highly suitable for BG/NBD and Gamma-Gamma spend heterogeneity modeling.
- **Source:** [KKBox Dataset on Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)

### B. **Sparkify Digital Music Service (Udacity/GitHub)**

- **Description:** Synthetic dataset simulating a music service with event-level logs.
- **Relevance:** Ideal for deep behavioral modeling and feature refinement.
- **Source:** [Sparkify Project Source](https://github.com/yduh/Churn-Prediction)

### C. **Telco Customer Churn (Kaggle/IBM)**

- **Description:** Classic dataset for contractual survival benchmarking.
- **Relevance:** Excellent for comparing statistical models with ML refiners.
- **Source:** [Telco Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## How CLTV is Computed

CLTV computation is split by business model, following the **"Economic Physics of Attention"** framework:

### 1. Non-Contractual Models (Gaming/TVOD)

We use the **BTYD (Buy 'Til You Die)** paradigm. It decomposes CLTV into two independent processes:
- **Transaction Process (BG/NBD Model):** Predicts how many transactions a customer will make in a future period. It uses the **RFM Vector** (Frequency $x$, Recency $t_x$, and Tenure $T$).
- **Monetary Process (Gamma-Gamma Model):** Predicts the average transaction value, assuming monetary value is independent of transaction frequency.
- **Computation:** $CLTV = E[\text{Transactions}] \times E[\text{Monetary Value}] \times \text{Margin}$

### 2. Contractual Models (SVOD/Subscriptions)

We use the **Survival Paradigm** to model revenue as the **Present Value of Expected Future Cash Flows**:
- **Survival Probability $S(t)$:** Calculated via **sBG** (discrete) or **Weibull AFT** (continuous). The latter incorporates **Engagement** signals as "accelerators" of the time-to-churn.
- **Revenue $R$:** The fixed subscription amount (ARPU).
- **Computation:** $CLTV = \sum_{t=1}^{n} \frac{S(t) \times R}{(1+d)^t}$ (where $d$ is the discount rate).

### 3. ML Refinement (XGBoost)

We employ **XGBoost as a Residual Refiner**. While statistical models capture the "Baseline Physics" of churn, XGBoost bridges the gap by modeling the residuals (errors) using high-dimensional behavioral signals:
- **QoE (Quality of Experience):** Captures buffering ratios and technical failures that trigger immediate churn.
- **Breadth & Velocity:** Analyzes catalog exploration and sudden changes in usage intensity.
- **Final Output:** $CLTV_{\text{Final}} = CLTV_{\text{Statistical}} + \Delta_{\text{XGBoost}}$. This ensures high precision without losing the interpretability of probabilistic models.

## Summary for your Project

1. **Load Data:** Ensure your DataFrame has `customer_id`, `transaction_date`, and `transaction_value`.
2. **Transform:** Use `lifetimes.utils.summary_data_from_transaction_data` to convert the raw logs into the **RFM** format (frequency, recency, T, monetary_value).
3. **Fit BG/NBD:** Use `frequency`, `recency`, and `T` to predict future transaction counts.
4. **Fit Gamma-Gamma:** Use `frequency` and `monetary_value` to predict average transaction value.
5. **Calculate CLV:** Combine both models to get the Customer Lifetime Value.

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
