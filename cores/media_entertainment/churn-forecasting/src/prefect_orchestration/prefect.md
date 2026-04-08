# Orchestration Layer (Prefect)

**Role:** DevOps Engineer (Distributed Workflows).

## 1. Lambda Architecture Execution
- **Speed Layer (Real-Time):** Prefect tasks triggered by Kafka/Webhook events for immediate payment failure retries and technical QoE alerts.
- **Batch Layer (Deep Learning):** Scheduled weekly flows for heavy CNN-BiLSTM retraining and 90-day sequence processing.

## 2. Resilience & Integration
- **Retries:** Mandatory for CDN metadata APIs and Payment Gateway hooks (Exponential backoff).
- **Kedro Integration:** 
    ```python
    @task
    def run_churn_node(pipeline_name: str):
        with KedroSession.create(project_path=".") as session:
            session.run(pipeline_name=pipeline_name)
    ```
- **Secrets:** Use `prefect.blocks` to manage API keys for competitive pricing feeds and content metadata.

## 3. Observability
- Monitor **Concept Drift** tasks; if AUC-ROC drops below 0.90, trigger an emergency "Model Audit" flow.