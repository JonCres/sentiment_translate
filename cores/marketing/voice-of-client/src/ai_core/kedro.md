---
title: "PIPELINE CONTEXT: KEDRO DATA SCIENCE CORE"
description: "Architecture and standards for modular data engineering and data science pipelines."
audience: developer
doc-type: reference
last-updated: 2026-02-13
---

# PIPELINE CONTEXT: KEDRO DATA SCIENCE CORE

## 1. Pipeline Architecture
The Kedro project is organized into four main pipelines, with `data_science` being a unified modular engine.

### A. `data_processing` (Ingestion & Cleaning)
*   **Purpose:** Raw data ingestion and sanitization.
*   **Mandatory Step:** **PII Redaction**. Uses Microsoft Presidio to scrub PII from `feedback_text`.
*   **Feature Engineering:** Generates the `04_feature` layer for downstream models.

### B. `data_science` (The Intelligence Engine)
A unified pipeline combining multiple modular sub-pipelines:
*   **ABSA (`absa`):** Aspect-Based Sentiment Analysis using PyABSA.
*   **Emotion Recognition (`mer`):** Multimodal Emotion Recognition for detecting client sentiment.
*   **Sentiment Scoring (`sentiment_scoring`):** Context-aware sentiment polarity and intensity.
*   **Ratings Prediction (`ratings_prediction`):** Predictive scoring for CSAT/NPS based on text.
*   **Topic Modeling (`topic_modeling`):** Unsupervised theme discovery using BERTopic.
*   **Churn Prediction (`churn_prediction`):** Gradient Boosting models (XGBoost/LightGBM) to identify at-risk accounts.
*   **Analysis (`analysis`):** Synthesis of model outputs for dashboard consumption.

### C. `monitoring` (MLOps & Governance)
*   **Purpose:** Drift detection and performance monitoring.
*   **Integration:** Uses `Evidently AI` for distribution checks and `MLflow` for experiment tracking.
*   **Quality Gate:** Validates model performance against business-defined thresholds.

### D. `visualization` (Presentation Data)
*   **Purpose:** Pre-calculates data structures for the Streamlit dashboard.
*   **Output:** Aggregated metrics and plot-ready datasets in the `08_reporting` layer.

## 2. Data Catalog Standards
*   **Layering:** Follows the Medallion Architecture (Raw -> Intermediate -> Primary -> Feature -> Model Input -> Models -> Model Output -> Reporting).
*   **Formats:** Uses `PolarsDeltaDataset` for high-performance tabular data and `CloudPickleDataset` for model artifacts.
*   **Versioning:** Mandatory versioning for all datasets in `04_feature` and above.

## 3. Quality Gates & Execution
*   **Tagging:** Pipelines use tags (e.g., `absa`, `churn`) for granular execution control.
*   **Registry:** All pipelines are registered in `src/ai_core/pipeline_registry.py`.
*   **Environment:** Execution is managed via `uv run kedro run`.
