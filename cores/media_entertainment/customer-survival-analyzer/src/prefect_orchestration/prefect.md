# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Workflow Orchestration
- **Real-Time Layer:** Prefect tasks triggered by Kafka/Flink for daily hazard function updates and immediate intervention triggers.
- **Batch Layer:** Weekly flows for cohort-level Kaplan-Meier updates and global feature importance rankings.

## 2. Resilience & Integration
- **Retries:** Mandatory exponential backoff for Payment Processor APIs and Competitive Intelligence feeds.
- **Kedro Integration:** 
    ```python
    @task
    def run_survival_node(pipeline_name: str):
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` for Billing platform credentials and CDN telemetry access.

## 3. Observability
- **Drift Gate:** Monitor the **Feature Stability Index**. If behavioral hazard drivers shift >0.10 monthly, trigger an emergency `[MODE: DEBUG]` alert.
- **SLA Monitoring:** Ensure daily hazard recalculations complete by 2AM UTC to meet intervention windows.