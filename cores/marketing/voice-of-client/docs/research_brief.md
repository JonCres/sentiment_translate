---
title: "Research Brief: Voice of the Client Project Documentation"
description: "Inventory of project components, audience identification, and gap detection for documentation engineering."
audience: developer
doc-type: reference
last-updated: 2026-02-13
---

# Research Brief: Voice of the Client Project Documentation

## Project Inventory
- **Core Technology:** Kedro (Pipelines), Prefect (Orchestration), Streamlit (UI), UV (Package Manager).
- **Domain:** B2B Strategic Account Intelligence (Marketing/Client Feedback).
- **Core Pipelines:**
    - `data_processing`: Cleaning and PII redaction.
    - `data_science`: Unified pipeline including ABSA, MER, Sentiment Scoring, Ratings Prediction, Topic Modeling, Churn Prediction, and Analysis.
    - `visualization`: Data for dashboards.
    - `monitoring`: Drift detection.
- **Orchestration:** Prefect flows for Data, AI, Monitoring, and Visualization pipelines.

## Target Audience
- **Data Scientists/Engineers:** Need to understand pipeline structure, Kedro nodes, and model integration.
- **DevOps/MLOps:** Need to understand Prefect orchestration, deployment, and monitoring.
- **Business Stakeholders:** Need to understand the value proposition, data requirements, and dashboard usage.

## Identified Gaps & Inconsistencies
1. **README Inconsistencies:**
    - Placeholders like `[INSERT_DATA_FOCUS_HERE]` and `[DATASET_NAME_1]` remain unfilled.
    - Streamlit entry point is listed as `app/main.py` but is actually `app/app.py`.
2. **Orchestration Mismatch:**
    - `prefect.md` describes conceptual flows (`ingest_and_assess_flow`, etc.) that do not match the implemented Python files (`data_pipeline.py`, `ai_pipeline.py`, etc.).
3. **Kedro Pipeline Naming:**
    - `kedro.md` uses different names for pipelines (`nlp_processing`, `predictive_modeling`) than those registered in `pipeline_registry.py` and implemented in `src/ai_core/pipelines/`.
4. **Architecture Documentation:**
    - The "Hybrid Architecture" (Prefect + Kedro) is introduced in `README.md` and `technical_design.md` but could benefit from a centralized, clear explanation.

## Sources of Truth
- `src/ai_core/pipeline_registry.py`: Definitive list of Kedro pipelines.
- `src/ai_core/pipelines/`: Actual implementation of nodes and pipelines.
- `src/prefect_orchestration/`: Actual implementation of Prefect flows.
- `app/app.py`: Streamlit entry point.
- `root_context.md`: Project's strategic mandate.
