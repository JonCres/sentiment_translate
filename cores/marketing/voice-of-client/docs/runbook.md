---
title: "Operational Runbook: voice-of-client"
description: "Support and maintenance procedures for on-call engineers managing the voice-of-client system."
audience: operator
doc-type: how-to
last-updated: 2026-02-13
---

# Operational Runbook: voice-of-client

**For Support Teams & On-Call Engineers**  
**Version:** 1.0.0 | **Last Updated:** 2026-02-13

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Deployment Architecture](#2-deployment-architecture)
3. [Monitoring & Health Checks](#3-monitoring--health-checks)
4. [Common Operations](#4-common-operations)
5. [Troubleshooting Guide](#5-troubleshooting-guide)
6. [Incident Response](#6-incident-response)
7. [Maintenance Procedures](#7-maintenance-procedures)
8. [Escalation Paths](#8-escalation-paths)

---

## 1. System Overview

### 1.1 Purpose
**voice-of-client** is a production AI system that transforms unstructured B2B feedback (NPS, Support, CRM) into predictive churn risk intelligence and actionable revenue-protection strategies. It processes multi-channel client feedback and serves predictions to Account Managers and Marketing teams.

### 1.2 Key Components

| Component | Technology | Purpose | Location |
|-----------|-----------|---------|----------|
| **Data Pipelines** | Kedro 0.19+ | ETL, feature engineering, model training | `src/ai_core/pipelines/` |
| **Orchestration** | Prefect 2.x/3.x | Workflow scheduling and monitoring | `src/prefect_orchestration/` |
| **Feature Store** | Feast 0.38+ | Feature versioning and serving | `feature_repo/` |
| **Model Registry** | MLflow | Model versioning and deployment | MLflow Server |
| **Dashboard** | Streamlit | Business user interface | `app/app.py` |
| **Data Storage** | S3/Local/DB | Raw data, features, predictions | `conf/base/catalog.yml` |

### 1.3 Service Dependencies

```
┌─────────────────┐
│  Streamlit App  │ (Port 8501)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ Kedro Pipelines │────▶│ Feast Server │ (Port 6566)
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ Prefect Server  │────▶│ MLflow Server│ (Port 5000)
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│ Data Sources    │ (S3, PostgreSQL, APIs)
└─────────────────┘
```

---

## 2. Deployment Architecture

### 2.1 Environment Configuration

| Environment | Purpose | Access | Data Refresh |
|-------------|---------|--------|--------------|
| **Development** | Local testing, experimentation | All developers | Manual |
| **Staging** | Pre-production validation | QA team, DevOps | Daily (6 AM UTC) |
| **Production** | Live system serving business users | Business users, API clients | `[FREQUENCY]` |

### 2.2 Service Endpoints

**Production:**
- **Streamlit Dashboard:** `https://dashboard.example.com`
- **MLflow UI:** `https://mlflow.example.com`
- **Prefect UI:** `https://prefect.example.com`
- **Feast Server:** `feast.example.com:6566`
- **API:** `https://api.example.com/v1`

**Staging:**
- **Streamlit Dashboard:** `https://staging-dashboard.example.com`
- **MLflow UI:** `https://staging-mlflow.example.com`

### 2.3 Credentials & Secrets

**Location:** `conf/local/credentials.yml` (gitignored)

**Required Secrets:**
```yaml
# AWS S3 (if using cloud storage)
aws:
  access_key_id: ${AWS_ACCESS_KEY_ID}
  secret_access_key: ${AWS_SECRET_ACCESS_KEY}
  region: us-east-1

# Database (if applicable)
database:
  host: ${DB_HOST}
  port: 5432
  username: ${DB_USER}
  password: ${DB_PASSWORD}
  database: ${DB_NAME}

# MLflow
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}
  registry_uri: ${MLFLOW_REGISTRY_URI}

# API Keys (if applicable)
api_keys:
  external_service: ${EXTERNAL_API_KEY}
```

**Secret Management:**
- **Development:** Local `.env` file (never commit!)
- **Production:** Prefect Blocks or cloud secret manager (AWS Secrets Manager, GCP Secret Manager)

---

## 3. Monitoring & Health Checks

### 3.1 System Health Dashboard

**Access:** Prefect UI → Flows → `[ai_core_name]_health_check`

**Key Metrics:**

| Metric | Healthy Threshold | Warning | Critical |
|--------|------------------|---------|----------|
| **Pipeline Success Rate** | >95% | 90-95% | <90% |
| **Inference Latency (p95)** | <100ms | 100-200ms | >200ms |
| **Data Freshness** | <24 hours | 24-48 hours | >48 hours |
| **Model Drift (PSI)** | <0.1 | 0.1-0.25 | >0.25 |
| **Feature Store Lag** | <5 minutes | 5-15 minutes | >15 minutes |
| **Dashboard Uptime** | 99.9% | 99.0-99.9% | <99.0% |

### 3.2 Manual Health Checks

#### **Check 1: Verify Pipeline Execution**
```bash
# SSH into the server or use Prefect UI
cd /path/to/ai_core

# Check last pipeline run status
prefect deployment run "[ai_core_name]/daily_pipeline"

# View logs
prefect flow-run logs <flow-run-id>
```

**Expected Output:** `Completed` status with no errors

---

#### **Check 2: Verify MLflow Model Registry**
```bash
# Open MLflow UI
open https://mlflow.example.com

# Or use CLI
mlflow models list --registered-model-name "[model_name]"
```

**Expected Output:** At least one model in `Production` stage

---

#### **Check 3: Verify Feast Feature Store**
```bash
# Check feature store status
cd feature_repo
feast feature-views list

# Test feature retrieval
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**Expected Output:** All feature views listed, materialization successful

---

#### **Check 4: Verify Streamlit Dashboard**
```bash
# Check if Streamlit is running
curl -I https://dashboard.example.com

# Or locally
ps aux | grep streamlit
```

**Expected Output:** HTTP 200 OK

---

### 3.3 Automated Alerts

**Alert Channels:**
- **Email:** `[ONCALL_EMAIL]`
- **Slack:** `#ai-core-alerts`
- **PagerDuty:** `[PAGERDUTY_SERVICE_KEY]`

**Alert Rules:**

| Alert | Trigger | Severity | Action |
|-------|---------|----------|--------|
| **Pipeline Failure** | 2 consecutive failures | 🔴 Critical | Investigate immediately |
| **Data Freshness** | No new data in 48 hours | 🟡 Warning | Check data sources |
| **Model Drift** | PSI > 0.25 | 🟡 Warning | Schedule model retraining |
| **High Latency** | p95 > 200ms for 10 minutes | 🟡 Warning | Check resource utilization |
| **Dashboard Down** | HTTP 5xx for 5 minutes | 🔴 Critical | Restart Streamlit service |

---

## 4. Common Operations

### 4.1 Starting Services

#### **Start Streamlit Dashboard**
```bash
cd /path/to/ai_core
source .venv/bin/activate  # or conda activate [env_name]

# Start dashboard
streamlit run app/app.py --server.port 8501
```

**Verify:** Open `http://localhost:8501` in browser

---

#### **Start MLflow Server**
```bash
cd /path/to/ai_core

# Start MLflow tracking server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

**Verify:** Open `http://localhost:5000` in browser

---

#### **Start Prefect Server (if self-hosted)**
```bash
# Start Prefect server
prefect server start

# In another terminal, start agent
prefect agent start -q default
```

**Verify:** Open `http://localhost:4200` in browser

---

### 4.2 Running Pipelines Manually

#### **Run Full Pipeline**
```bash
cd /path/to/ai_core
source .venv/bin/activate

# Run all pipelines
kedro run

# Or via Prefect
prefect deployment run "[ai_core_name]/daily_pipeline"
```

---

#### **Run Specific Pipeline**
```bash
# Run only data processing
kedro run --pipeline data_processing

# Run only model training
kedro run --pipeline data_science
```

---

#### **Run with Custom Parameters**
```bash
# Override parameters
kedro run --params "lookback_days=180,model_type=xgboost"
```

---

### 4.3 Model Deployment

#### **Promote Model to Production**
```bash
# Via MLflow UI
# 1. Navigate to Models → [model_name]
# 2. Select version → Transition to → Production

# Or via CLI
mlflow models transition \
  --name "[model_name]" \
  --version 5 \
  --stage Production
```

---

#### **Rollback to Previous Model**
```bash
# Transition current production model to Archived
mlflow models transition \
  --name "[model_name]" \
  --version 5 \
  --stage Archived

# Promote previous version to Production
mlflow models transition \
  --name "[model_name]" \
  --version 4 \
  --stage Production
```

---

### 4.4 Data Refresh

#### **Manual Data Refresh**
```bash
cd /path/to/ai_core

# Run data processing pipeline
kedro run --pipeline data_processing

# Materialize features to Feast
cd feature_repo
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

---

### 4.5 MLflow Operations

This AI Core integrates with **MLflow** for experiment tracking, model versioning, and monitoring metrics.

#### **MLflow Files**

The following MLflow files are included in the AI Core:

| File | Purpose |
|------|---------|
| `src/utils/mlflow_tracking.py` | Utility module for MLflow setup, run management, and Prefect linking |
| `conf/local/mlflow.yml` | Local MLflow config (credentials, custom tracking URI) - **GITIGNORED** |
| `conf/base/parameters.yml` | MLflow settings under `mlops:` section |

---

#### **Initial Setup After Cloning**

> **Note**: All commands should be run from the root of the AI Core folder.

```bash
# 1. The MLflow tracking utility is already included in src/utils/mlflow_tracking.py

# 2. Configure your MLflow tracking URI in conf/base/parameters.yml
# Default is http://localhost:5000

# 3. (Optional) Add credentials to conf/local/mlflow.yml for remote servers
# Note: conf/local/ is gitignored and should NOT be committed

# 4. Start local MLflow server (use uv run for proper environment)
uv run mlflow server --host 127.0.0.1 --port 5000

# 5. Run your pipeline - metrics will be logged automatically
uv run kedro run --pipeline=monitoring
```

**Verify:** Open `http://localhost:5000` in browser to access MLflow UI

---

#### **MLflow Features**

- **Experiment Tracking**: All pipeline runs log parameters, metrics, and artifacts automatically
- **Model Registry**: Trained models are versioned and can be promoted through stages (Staging → Production)
- **Prefect Integration**: Bi-directional linking between Prefect flows and MLflow runs for full traceability
- **Artifact Storage**: Model artifacts, plots, and data samples stored with each run
- **Metric Comparison**: Compare multiple runs side-by-side in the MLflow UI

---

#### **Configuration**

MLflow settings are configured in `conf/base/parameters.yml`:

```yaml
mlops:
  tracking_uri: "http://localhost:5000"
  experiment_name: "my_experiment"
  log_monitoring_metrics: true
  link_prefect_runs: true
  model_registry:
    enabled: true
    auto_register: true
```

**Configuration Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tracking_uri` | MLflow server URL | `http://localhost:5000` |
| `experiment_name` | Name of the MLflow experiment | `my_experiment` |
| `log_monitoring_metrics` | Enable automatic metric logging | `true` |
| `link_prefect_runs` | Link Prefect flow runs to MLflow | `true` |
| `model_registry.enabled` | Enable model registry | `true` |
| `model_registry.auto_register` | Auto-register trained models | `true` |

---

#### **Loading Models from Registry**

**Via Python API:**

```python
import mlflow

# Load production model
model = mlflow.sklearn.load_model("models:/regressor/Production")

# Load specific version
model = mlflow.sklearn.load_model("models:/regressor/3")

# Load model from specific run
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

**Via CLI:**

```bash
# List all registered models
mlflow models list

# Get model details
mlflow models describe --name "regressor"

# Download model artifacts
mlflow artifacts download --run-id <run_id> --dst-path ./models/
```

---

#### **Common MLflow Operations**

**View Experiment Runs:**
```bash
# Open MLflow UI
uv run mlflow ui --port 5000

# Or if MLflow server is already running
open http://localhost:5000
```

**Search Runs Programmatically:**
```python
import mlflow

# Search for best run by metric
runs = mlflow.search_runs(
    experiment_names=["my_experiment"],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)
print(runs)
```

**Log Custom Metrics:**
```python
from src.utils.mlflow_tracking import get_or_create_experiment

with mlflow.start_run(experiment_id=get_or_create_experiment("my_experiment")):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

---

#### **MLflow Server Management**

**Start MLflow Server (Development):**
```bash
# Using uv (recommended)
uv run mlflow server --host 127.0.0.1 --port 5000

# With custom backend store
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

**Start MLflow Server (Production):**
```bash
# With PostgreSQL backend and S3 artifact store
uv run mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-mlflow-bucket/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Check MLflow Server Status:**
```bash
# Check if server is running
curl http://localhost:5000/health

# Or
ps aux | grep mlflow
```

---

#### **Troubleshooting MLflow**

**Issue: MLflow server not accessible**

**Diagnosis:**
```bash
# Check if server is running
ps aux | grep mlflow

# Check port availability
lsof -i :5000
```

**Resolution:**
```bash
# Kill existing MLflow process
pkill -f "mlflow server"

# Restart MLflow server
uv run mlflow server --host 127.0.0.1 --port 5000
```

---

**Issue: Models not appearing in registry**

**Diagnosis:**
```bash
# Check MLflow configuration
cat conf/base/parameters.yml | grep mlops

# Verify model registration is enabled
```

**Resolution:**
1. Ensure `model_registry.enabled: true` in `conf/base/parameters.yml`
2. Check that models are being logged in pipeline code
3. Verify MLflow tracking URI is correct
4. Re-run the training pipeline:
   ```bash
   uv run kedro run --pipeline data_science
   ```

---

**Issue: Experiment tracking not working**

**Diagnosis:**
```bash
# Check MLflow tracking URI
echo $MLFLOW_TRACKING_URI

# Verify MLflow utility is imported
grep -r "mlflow_tracking" src/aicore/pipelines/
```

**Resolution:**
1. Ensure `MLFLOW_TRACKING_URI` environment variable is set (or configured in `parameters.yml`)
2. Verify `src/utils/mlflow_tracking.py` is being used in pipeline nodes
3. Check MLflow server logs for errors
4. Test connection:
   ```python
   import mlflow
   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   print(mlflow.get_tracking_uri())
   ```

---

## 5. Troubleshooting Guide

### 5.1 Pipeline Failures

#### **Symptom:** Pipeline fails with `FileNotFoundError`

**Diagnosis:**
```bash
# Check catalog configuration
cat conf/base/catalog.yml | grep [dataset_name]

# Verify file exists
ls -lh data/01_raw/[filename]
```

**Resolution:**
1. Verify data source is accessible (S3, database, API)
2. Check credentials in `conf/local/credentials.yml`
3. Ensure file path in `catalog.yml` is correct
4. Re-run data ingestion: `kedro run --pipeline data_processing --to-nodes [node_name]`

---

#### **Symptom:** Pipeline fails with `ModuleNotFoundError`

**Diagnosis:**
```bash
# Check installed packages
pip list | grep [package_name]

# Verify virtual environment
which python
```

**Resolution:**
```bash
# Reinstall dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

---

#### **Symptom:** Prefect task fails with "Pickling Error"

**Diagnosis:**
```bash
# Check Prefect logs
prefect flow-run logs <flow-run-id>
```

**Root Cause:** Attempting to pass non-serializable objects (KedroContext, DataCatalog) between tasks

**Resolution:**
1. Review `src/prefect_orchestration/flows.py`
2. Ensure tasks only pass configuration strings (pipeline names, dataset names)
3. Initialize `KedroSession` **inside** each Prefect task

**Correct Pattern:**
```python
@task
def run_pipeline_task(pipeline_name: str):
    with KedroSession.create(project_path=".") as session:
        session.run(pipeline_name=pipeline_name)
```

---

### 5.2 Model Performance Issues

#### **Symptom:** Model predictions are degraded (accuracy drop)

**Diagnosis:**
```bash
# Check model metrics in MLflow
mlflow ui

# Compare current vs. baseline metrics
# Navigate to: Experiments → [experiment_name] → Metrics
```

**Resolution:**
1. **Check for data drift:**
   ```bash
   kedro run --pipeline monitoring
   ```
2. **Retrain model with recent data:**
   ```bash
   kedro run --pipeline data_science
   ```
3. **Validate new model before deployment:**
   - Compare metrics in MLflow
   - Run A/B test if possible
4. **Promote to production if validated:**
   ```bash
   mlflow models transition --name "[model_name]" --version [new_version] --stage Production
   ```

---

#### **Symptom:** High inference latency (>200ms)

**Diagnosis:**
```bash
# Check resource utilization
top
htop

# Check Streamlit logs
tail -f streamlit.log
```

**Resolution:**
1. **Optimize caching:**
   - Ensure `@st.cache_resource` is used for model loading
   - Ensure `@st.cache_data` is used for DataFrames
2. **Reduce batch size:**
   - Limit dashboard queries to recent data (e.g., last 30 days)
3. **Scale resources:**
   - Increase CPU/RAM allocation
   - Use GPU for deep learning models

---

### 5.3 Dashboard Issues

#### **Symptom:** Streamlit dashboard shows "Connection Error"

**Diagnosis:**
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Check logs
tail -f streamlit.log
```

**Resolution:**
```bash
# Restart Streamlit
pkill -f streamlit
streamlit run app/app.py --server.port 8501
```

---

#### **Symptom:** Dashboard loads slowly or times out

**Diagnosis:**
```bash
# Check data volume
du -sh data/

# Check query performance
# Review Streamlit profiler in browser DevTools
```

**Resolution:**
1. **Implement pagination:**
   - Limit rows displayed (e.g., top 1000 customers)
2. **Optimize queries:**
   - Use Polars instead of Pandas for large datasets
   - Pre-aggregate data in pipelines
3. **Add caching:**
   ```python
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_predictions():
       return catalog.load("predictions")
   ```

---

### 5.4 Feature Store Issues

#### **Symptom:** Feast materialization fails

**Diagnosis:**
```bash
cd feature_repo
feast feature-views list

# Check feature store logs
feast serve --log-level DEBUG
```

**Resolution:**
1. **Verify feature definitions:**
   ```bash
   feast plan
   feast apply
   ```
2. **Check data sources:**
   - Ensure Kedro pipeline outputs are available
   - Verify file paths in `feature_store.yaml`
3. **Re-materialize:**
   ```bash
   feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
   ```

---

## 6. Incident Response

### 6.1 Incident Severity Levels

| Level | Definition | Response Time | Escalation |
|-------|-----------|---------------|------------|
| **P0 - Critical** | Production down, data loss, security breach | **15 minutes** | Immediate |
| **P1 - High** | Degraded performance, partial outage | **1 hour** | If unresolved in 2 hours |
| **P2 - Medium** | Non-critical feature broken, workaround exists | **4 hours** | If unresolved in 8 hours |
| **P3 - Low** | Minor issue, cosmetic bug | **Next business day** | N/A |

### 6.2 Incident Response Checklist

**For P0/P1 Incidents:**

- [ ] **1. Acknowledge** (within 15 minutes)
  - Post in `#ai-core-incidents` Slack channel
  - Update status page (if applicable)

- [ ] **2. Assess Impact**
  - How many users affected?
  - What functionality is broken?
  - Is data at risk?

- [ ] **3. Mitigate**
  - Implement immediate workaround (e.g., rollback model, restart service)
  - Communicate to stakeholders

- [ ] **4. Investigate Root Cause**
  - Review logs (Prefect, MLflow, Streamlit)
  - Check recent changes (git log, deployments)

- [ ] **5. Resolve**
  - Apply permanent fix
  - Validate in staging before production

- [ ] **6. Post-Mortem**
  - Document incident in `docs/incidents/[date]_incident.md`
  - Identify preventive measures
  - Update runbook

---

### 6.3 Rollback Procedures

#### **Rollback Model**
```bash
# Identify current production model
mlflow models list --registered-model-name "[model_name]"

# Transition to previous version
mlflow models transition \
  --name "[model_name]" \
  --version [previous_version] \
  --stage Production
```

---

#### **Rollback Code**
```bash
# Identify last stable commit
git log --oneline -10

# Revert to previous commit
git revert [commit_hash]
git push origin main

# Redeploy
# (Follow your deployment procedure)
```

---

#### **Rollback Data**
```bash
# Restore from backup (if versioned in S3)
aws s3 cp s3://bucket/backups/[date]/ data/ --recursive

# Re-run pipeline with restored data
kedro run --pipeline data_processing
```

---

## 7. Maintenance Procedures

### 7.1 Model Retraining Schedule

**Frequency:** `[e.g., Weekly, Monthly, Quarterly]`

**Procedure:**
1. **Trigger retraining:**
   ```bash
   kedro run --pipeline data_science
   ```
2. **Validate new model:**
   - Compare metrics in MLflow
   - Run holdout validation
3. **A/B test (if applicable):**
   - Deploy to 10% of traffic
   - Monitor for 48 hours
4. **Promote to production:**
   ```bash
   mlflow models transition --name "[model_name]" --version [new_version] --stage Production
   ```

---

### 7.2 Data Cleanup

**Frequency:** `[e.g., Monthly]`

**Procedure:**
```bash
# Archive old data (>90 days)
cd data/
tar -czf archive_$(date +%Y%m%d).tar.gz 01_raw/ 02_intermediate/
mv archive_*.tar.gz backups/

# Delete archived data
rm -rf 01_raw/* 02_intermediate/*

# Verify catalog still works
kedro catalog list
```

---

### 7.3 Dependency Updates

**Frequency:** `[e.g., Quarterly]`

**Procedure:**
```bash
# Update dependencies
uv pip install --upgrade -e .

# Run tests
pytest tests/

# Validate pipelines
kedro run --pipeline data_processing

# If successful, commit
git add pyproject.toml uv.lock
git commit -m "chore: update dependencies"
git push
```

---

## 8. Escalation Paths

### 8.1 On-Call Rotation

| Role | Primary | Backup | Contact |
|------|---------|--------|---------|
| **MLOps Engineer** | [Name] | [Name] | [Phone/Slack] |
| **Data Scientist** | [Name] | [Name] | [Phone/Slack] |
| **Tech Lead** | [Name] | [Name] | [Phone/Slack] |
| **Product Owner** | [Name] | [Name] | [Phone/Slack] |

### 8.2 Escalation Matrix

| Issue Type | First Contact | Escalate To | Escalate If |
|------------|--------------|-------------|-------------|
| **Pipeline Failure** | MLOps Engineer | Data Scientist | Unresolved in 2 hours |
| **Model Performance** | Data Scientist | Tech Lead | Accuracy drop >10% |
| **Dashboard Down** | MLOps Engineer | Tech Lead | Unresolved in 1 hour |
| **Data Quality** | Data Scientist | Product Owner | Business impact |
| **Security Incident** | Tech Lead | CISO | Any security breach |

---

## 9. Appendix

### 9.1 Useful Commands Cheat Sheet

```bash
# Kedro
kedro run                                    # Run all pipelines
kedro run --pipeline [name]                  # Run specific pipeline
kedro catalog list                           # List all datasets
kedro viz                                    # Visualize pipeline

# Prefect
prefect deployment run "[name]"              # Trigger deployment
prefect flow-run logs <id>                   # View logs
prefect agent start -q default               # Start agent

# MLflow
mlflow ui                                    # Start UI
mlflow models list                           # List models
mlflow models transition --name [name] --version [v] --stage Production

# Feast
feast feature-views list                     # List feature views
feast materialize-incremental [timestamp]    # Materialize features
feast serve                                  # Start feature server

# Streamlit
streamlit run app/app.py                     # Start dashboard
streamlit cache clear                        # Clear cache
```

---

### 9.2 Log Locations

| Component | Log Location |
|-----------|-------------|
| **Kedro** | `logs/journals/` |
| **Prefect** | Prefect UI → Flow Runs → Logs |
| **MLflow** | `mlruns/` |
| **Streamlit** | `streamlit.log` (if configured) |
| **System** | `/var/log/syslog` (Linux) or `Console.app` (Mac) |

---

### 9.3 Key Configuration Files

| File | Purpose |
|------|---------|
| `conf/base/catalog.yml` | Data sources and destinations |
| `conf/base/parameters.yml` | Model hyperparameters |
| `conf/local/credentials.yml` | Secrets (gitignored) |
| `pyproject.toml` | Python dependencies |
| `feature_repo/feature_store.yaml` | Feast configuration |

---

**Document Control:**
- **Review Cycle:** Quarterly or after major incidents
- **Approval Required:** Tech Lead, MLOps Lead
- **Related Documents:** `technical_design.md`, `user_guide.md`, `api_specification.md`

---

**Emergency Contacts:**
- **On-Call Engineer:** `[PHONE]`
- **Tech Lead:** `[PHONE]`
- **Slack:** `#ai-core-incidents`
