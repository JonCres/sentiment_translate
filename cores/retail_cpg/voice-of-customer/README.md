# Retail CPG: Voice of Customer 🚀

## Table of Contents

- [Retail CPG: Voice of Customer 🚀](#retail-cpg-voice-of-customer-)
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
  - [1. Ideal Data Profile: **Unstructured Customer Feedback**](#1-ideal-data-profile-unstructured-customer-feedback)
  - [2. Mandatory Variables \& Schema](#2-mandatory-variables--schema)
  - [3. Recommended Benchmark Datasets](#3-recommended-benchmark-datasets)
    - [A. **Amazon Product Data (E-commerce Sentiment)**](#a-amazon-product-data-e-commerce-sentiment)
    - [B. **Online Retail II (Transactional Context)**](#b-online-retail-ii-transactional-context)
    - [C. **MELD (Multimodal Emotion Recognition)**](#c-meld-multimodal-emotion-recognition)
    - [D. **Twitter/X Customer Support (Social Listening)**](#d-twitterx-customer-support-social-listening)
  - [How Customer Voice is Analyzed](#how-customer-voice-is-analyzed)
    - [1. Text Intelligence Layer (ABSA)](#1-text-intelligence-layer-absa)
    - [2. Multimodal Emotion Layer (MER)](#2-multimodal-emotion-layer-mer)
    - [3. Behavioral Context Layer](#3-behavioral-context-layer)
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

## 1. Ideal Data Profile: **Unstructured Customer Feedback**

The ideal dataset for this AI Core involves **multi-channel feedback** where **customers** exhibit **sentiment toward specific aspects** toward specific **product quality, shipping speed, service**.

**Why this fits:**

- **Granularity:** Customers often mention multiple aspects in a single review (e.g., "Great dress, but the delivery was slow"), which suits Aspect-Based Sentiment Analysis (ABSA).
- **Affective Nuance:** Audio and video logs capture emotions that text alone might miss, ideal for Multimodal Emotion Recognition (MER).
- **Temporal Depth:** Historical logs should span at least **6-12 months** to account for seasonality and trend shifts.

---

## 2. Mandatory Variables & Schema

To successfully execute the core pipelines, your input dataset should contain the following variables:

| Variable Name             | Description                                                     | Data Type      |
| :------------------------ | :-------------------------------------------------------------- | :------------- |
| **`interaction_id`**      | Unique identifier for the feedback event (review, call, tweet). | `str`          |
| **`interaction_payload`** | The raw content (text body, audio file path, or video link).    | `str` / `blob` |
| **`customer_id`**         | Unique ID to link interactions across channels and time.        | `str`          |
| **`timestamp`**           | The exact date and time the interaction occurred.               | `ISO 8601`     |
| **`target_object_id`**    | Identifier for the product (SKU) or store location.             | `str`          |

---

## 3. Recommended Benchmark Datasets

The following public datasets are recommended for establishing a baseline for this AI Core:

### A. **Amazon Product Data (E-commerce Sentiment)**

- **Description:** A massive corpus (233M+ reviews) covering electronics, apparel, and groceries.
- **Relevance:** The gold standard for pre-training retail ABSA models.
- **Source:** [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)

### B. **Online Retail II (Transactional Context)**

- **Description:** Transactional logs for identifying customer segments (High-Value, At-Risk) to prioritize insights.
- **Relevance:** Provides the necessary scale and behavioral context for retail analytics.
- **Source:** [Online Retail II on UCI](https://archive.ics.uci.edu/ml/datasets/online+retail+II)

### C. **MELD (Multimodal Emotion Recognition)**

- **Description:** Audio/Visual/Text dialogue for training emotion alignment.
- **Relevance:** Essential for fine-tuning multimodal models.
- **Source:** [MELD on GitHub](https://github.com/declare-lab/MELD)

### D. **Twitter/X Customer Support (Social Listening)**

- **Description:** Real-time brand interactions for high-urgency sentiment analysis.
- **Relevance:** Captures the high-velocity, low-latency nature of social media feedback.
- **Source:** [Social Media Sentiment on Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)

## How Customer Voice is Analyzed

Voice of Customer analysis is handled via a multi-layered composable AI approach:

### 1. Text Intelligence Layer (ABSA)

We use **Transformer-based ABSA (BERT/RoBERTa)** models fine-tuned on retail corpora to extract (Target, Aspect, Sentiment) triplets. This moves beyond document-level sentiment to understand that a review can be "Positive" about Style but "Negative" about Price.

### 2. Multimodal Emotion Layer (MER)

A fusion architecture that combines **1D-CNNs** (for acoustic features like pitch/jitter) and **3D-CNNs** (for visual facial cues) with text embeddings. This detects vocal stress or frustration even before customers explicitly complain.

### 3. Behavioral Context Layer

**Session-based Transformers** encode clickstream sequences (e.g., rage clicks, repeated cart abandonment) to generate "Implicit Sentiment Scores" for the 95% of 'silent' customers who never write reviews but simply churn.

## Summary for your Project

1. **Ingest Content:** Load reviews or call logs into the bronze data layer.
2. **Clean & Tokenize:** Run the `data_processing` pipeline to prepare text for transformer models.
3. **Analyze Aspects:** Execute `data_science` to extract specific sentiment for price, quality, etc.
4. **Visualize:** Use the Streamlit app to explore holistic customer health.

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
