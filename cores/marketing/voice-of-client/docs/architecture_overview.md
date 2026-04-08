---
title: "Architecture Overview: Hybrid Kedro-Prefect System"
description: "High-level design of the integrated Kedro transformation and Prefect orchestration layers."
audience: developer
doc-type: explanation
last-updated: 2026-02-13
---

# Architecture Overview: Hybrid Kedro-Prefect System

## 1. Executive Summary
The **Voice of the Client** AI Core utilizes a hybrid architecture that combines the data engineering excellence of **Kedro** with the industrial-grade orchestration of **Prefect**. This design ensures that data pipelines are reproducible, modular, and observable in a production environment.

## 2. Core Components

### 2.1 Transformation Layer (Kedro)
Kedro serves as the "engine room" where data transformations and model logic reside.
- **Pipelines:** Defined in `src/ai_core/pipelines/`, these are directed acyclic graphs (DAGs) of pure Python functions.
- **Data Catalog:** Managed via `conf/base/catalog.yml`, abstracting data I/O and ensuring consistency across environments.
- **Modularity:** The `data_science` pipeline is a composition of several specialized sub-pipelines (ABSA, Churn, etc.).

### 2.2 Orchestration Layer (Prefect)
Prefect acts as the "mission control," managing the execution of Kedro pipelines.
- **Flows:** Python scripts in `src/prefect_orchestration/` that wrap Kedro execution.
- **Observability:** Provides real-time monitoring, retries, and failure alerts.
- **Hybrid Parallelism:** The `complete_flow` executes `visualization` and `monitoring` pipelines in parallel using Prefect's concurrency primitives.

### 2.3 Feature Management (Feast)
- **Feature Store:** Centralized repository for versioned features.
- **Consistency:** Ensures that the same feature definitions are used for both training (offline) and inference (online).

### 2.4 Experiment Tracking (MLflow)
- **Tracking:** Logs parameters, metrics, and artifacts during pipeline runs.
- **Model Registry:** Managed lifecycle for trained models (ABSA, Churn).

## 3. Data Flow Architecture

1.  **Ingestion:** Raw data arrives in the `01_raw` layer.
2.  **Processing:** `data_flow` (Prefect) triggers the `data_processing` pipeline (Kedro), resulting in the `04_feature` layer.
3.  **Intelligence:** `ai_flow` (Prefect) triggers the `data_science` pipeline (Kedro), generating models and predictions in the `07_model_output` layer.
4.  **Verification:** `monitoring_flow` (Prefect) checks for drift and performance via Kedro.
5.  **Reporting:** `visualization_flow` (Prefect) prepares data for the Streamlit app.
6.  **Presentation:** The Streamlit dashboard (`app/app.py`) reads from the `08_reporting` layer to display insights.

## 4. Deployment Model
- **Package Management:** Managed by `uv` for reproducible environments.
- **Execution:** All components are executed via `uv run` to ensure dependency alignment.
- **Infrastructure:** Designed for containerized deployment (Docker) on Kubernetes or cloud-native orchestration platforms.
