# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Multi-Speed Execution
- **Speed Layer (Streaming):** Prefect tasks triggered by Kafka for sub-second feature updates (e.g., real-time ad-load adjustments).
- **Batch Layer (Training):** Weekly flows for content embedding updates and full ensemble retraining.

## 2. Resilience & Integration
- **Retries:** Mandatory exponential backoff for external APIs (Sports calendars, CPM benchmarks, Macro indicators).
- **Kedro Integration:** 
    ```python
    @task
    def run_cltv_node(pipeline_name: str):
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` for Billing API keys and Ad Server credentials.

## 3. Observability
- **Drift Gate:** Monitor the **Population Stability Index (PSI)**. If feature drift exceeds 0.20, block automated deployment and trigger an alert to the CRO/MLOps lead.
- **SLA Monitoring:** Ensure real-time inference latency remains <100ms for personalization engine calls.