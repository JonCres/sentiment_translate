# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Workflow Orchestration
- **Batch Layer:** Weekly flows for full cohort re-scoring and CLV distribution updates.
- **Near-Real-Time Layer:** Daily tasks for engagement recency updates and high-urgency intervention triggers.

## 2. Resilience & Integration
- **Retries:** Mandatory exponential backoff for external enrichment APIs (Clearbit) and Billing systems.
- **Kedro Integration (The Pickle Rule):** 
    ```python
    @task
    def run_clv_scoring(pipeline_name: str):
        # Initialize session inside the task to avoid serialization issues
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` for database credentials and API gateway keys.

## 3. Observability
- **Drift Gate:** Monitor **Population Stability Index (PSI)**. If behavioral distributions shift >0.20, block automated deployment and trigger an alert to the CRO/MLOps lead.
- **Freshness:** Alert if feature freshness exceeds 24 hours for active engagement aggregates.