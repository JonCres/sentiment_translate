---
title: "ORCHESTRATION CONTEXT: PREFECT WORKFLOWS"
description: "Deployment and management of Kedro pipelines using Prefect orchestration."
audience: operator
doc-type: reference
last-updated: 2026-02-13
---

# ORCHESTRATION CONTEXT: PREFECT WORKFLOWS

## 1. Workflow Philosophy
Prefect is responsible for the **reliability, timing, and observability** of the AI Core. It orchestrates the Kedro pipelines defined in `src/ai_core`.

## 2. Implemented Flows

### Flow A: `data_flow` (Data Pipeline)
*   **Source:** `src/prefect_orchestration/data_pipeline.py`
*   **Purpose:** Orchestrates the Kedro `data_processing` pipeline.
*   **Steps:** Extract, Validate, Clean, and Feature Engineering.

### Flow B: `ai_flow` (AI Pipeline)
*   **Source:** `src/prefect_orchestration/ai_pipeline.py`
*   **Purpose:** Orchestrates the Kedro `data_science` pipeline.
*   **Steps:** Train, Evaluate, and Register models (ABSA, Sentiment, Churn, etc.).

### Flow C: `monitoring_flow` (Monitoring Pipeline)
*   **Source:** `src/prefect_orchestration/monitoring_pipeline.py`
*   **Purpose:** Orchestrates the Kedro `monitoring` pipeline with MLflow integration.
*   **Steps:** Drift Detection and Performance Checks.

### Flow D: `visualization_flow` (Visualization Pipeline)
*   **Source:** `src/prefect_orchestration/visualization_pipeline.py`
*   **Purpose:** Orchestrates the Kedro `visualization` pipeline.
*   **Steps:** Generate charts and plots for the dashboard.

### Flow E: `complete_flow` (Master Pipeline)
*   **Source:** `src/prefect_orchestration/run_all_pipelines.py`
*   **Purpose:** Master orchestration of all pipelines.
*   **Sequence:**
    1.  Execute `data_flow`.
    2.  Execute `ai_flow`.
    3.  Execute `visualization_flow` and `monitoring_flow` in parallel.

## 3. Governance & MLflow Integration
*   **MLflow Tracking:** The `monitoring_flow` is explicitly linked with MLflow experiments. Prefect run IDs are linked to MLflow runs for end-to-end traceability.
*   **Config Management:** Flows use a centralized `configs/project_config.yaml` loaded via a utility.
*   **Deployment:** Flows are deployed using Prefect's `deploy` function with `from_source` to handle code loading.

## 4. Operational Guardrails
*   **PII Check:** The `data_processing` pipeline (invoked by `data_flow`) contains mandatory PII redaction nodes.
*   **Secrets:** API Keys and credentials must be accessed via Prefect Blocks or environment variables, managed by the orchestration layer.
