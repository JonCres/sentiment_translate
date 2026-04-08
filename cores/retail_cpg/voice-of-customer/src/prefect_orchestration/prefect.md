# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Lambda Architecture Execution
- **Real-Time Layer:** Prefect tasks triggered by Kafka streams for <60s sentiment alerts and dynamic IVR routing.
- **Batch Layer:** Daily/Weekly flows for LDA topic discovery, competitive benchmarking, and model retraining.

## 2. Resilience & Integration
- **Kedro Integration:** 
    ```python
    @task
    def run_voc_node(pipeline_name: str):
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` for PIM/ERP credentials and API keys.

## 3. Observability
- **Drift Gate:** Monitor "Sentiment Velocity." If distribution shifts >10%, trigger an emergency retraining flow.
- **SLA Monitoring:** Alert if processing latency (p95) exceeds 2 seconds for real-time channels.