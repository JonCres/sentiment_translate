# AI Cores System Architecture

This document provides a high-level overview of the architectural design of the AI Cores platform, explaining how various technologies integrate to deliver production-ready AI solutions.

## 1. The Unified Stack

The AI Cores platform utilizes a hybrid architecture that combines best-in-class tools for different stages of the ML lifecycle:

| Layer | Technology | Responsibility |
| :--- | :--- | :--- |
| **Orchestration** | **Prefect 3.x** | Workflow management, scheduling, retries, and observability. |
| **Transformation** | **Kedro 0.19+** | Data engineering pipelines, modular nodes, and reproducible transformations. |
| **Experimentation**| **MLflow** | Experiment tracking, model registry, and artifact management. |
| **Feature Store** | **Feast** | Centralized feature management and low-latency serving. |
| **Packaging** | **uv** | Fast dependency management and hardware-accelerated environments. |
| **Presentation** | **Streamlit** | Interactive dashboards and AI-driven insights. |

## 2. Integrated Pipeline Flow

Every AI Core follows a standardized multi-phase workflow that ensures consistency and quality.

### Phase 1: Data Processing (Kedro)
- **Ingestion**: Reading from source systems (S3, SQL, API).
- **Validation**: Schema enforcement using **Pandera**.
- **Cleaning**: Handling missing values and outliers.
- **Feature Engineering**: Creating domain-specific signals.
- **Feast Sync**: Pushing features to the Feature Store.

### Phase 2: Data Science (Kedro + MLflow)
- **Training**: Executing ML models (Random Forest, DeepSurv, etc.).
- **Evaluation**: Calculating performance metrics (AUC, RMSE).
- **Registry**: Automatically logging models and metrics to the **MLflow Model Registry**.

### Phase 3: Orchestration (Prefect)
Prefect acts as the "brain" that triggers Kedro pipelines. It handles:
- **Scheduling**: Running pipelines on a regular basis.
- **Infrastructure**: Executing tasks in Docker or Kubernetes.
- **Observability**: Providing a unified dashboard for all pipeline runs.

## 3. The "Pickle Rule" & Interoperability

To ensure that Prefect can orchestrate Kedro without serialization issues, we follow the **Pickle Rule**:
> **NEVER** pass live Python objects (like Kedro contexts or active DB connections) between Prefect tasks. Instead, pass configuration keys and re-initialize the session inside each worker.

## 4. Hardware Acceleration

The platform is designed to run anywhere, from local MacBooks to high-performance GPU clusters. This is handled via `uv` extras:
- `uv sync --extra mps`: Optimized for Apple Silicon.
- `uv sync --extra cuda`: Optimized for NVIDIA GPUs.
- `uv sync --extra xpu`: Optimized for Intel GPUs.

## 5. Security & Secret Management

- **Credentials**: Managed via Prefect Blocks (Secrets) or Kedro's `credentials.yml` (local only).
- **Data Privacy**: Mandatory PII masking and hashing at the ingestion layer.
