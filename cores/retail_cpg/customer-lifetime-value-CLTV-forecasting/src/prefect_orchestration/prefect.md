# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Hybrid Processing SLA
- **Transactional Batch:** Daily flow with 24-hour SLA for POS/ERP ingestion.
- **Behavioral Stream:** Real-time Prefect tasks (15-minute latency) for engagement decay detection and P(Churn) updates.

## 2. Resilience & Integration
- **Retries:** Mandatory exponential backoff for external enrichment APIs (Clearbit, OpenWeather).
- **Kedro Integration:** 
    ```python
    @task
    def run_clv_prediction(pipeline_name: str):
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` for database credentials and API keys.

## 3. Observability
- **Drift Gate:** If Feature Drift (PSI) > 0.25, block the automated deployment and trigger an `[MODE: DEBUG]` alert to the MLOps team.
- **Freshness:** Monitor "Feature Staleness"; alert if >5% of active customers haven't been re-scored in 72 hours.