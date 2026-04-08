# Model Monitoring & Drift Detection

**Project:** Predictive CLTV Insights
**Industry:** Media & Entertainment
**Last Updated:** 2026-02-13

---

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Drift Detection](#drift-detection)
4. [Performance Monitoring](#performance-monitoring)
5. [Alert Configuration](#alert-configuration)
6. [Monitoring Dashboard](#monitoring-dashboard)
7. [Model Retraining Triggers](#model-retraining-triggers)
8. [Incident Response](#incident-response)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The CLTV prediction system implements comprehensive monitoring to detect:

- **Data drift**: Changes in input feature distributions
- **Concept drift**: Changes in customer behavior patterns (target distribution)
- **Model performance degradation**: Declining prediction accuracy
- **System health**: Pipeline failures, latency issues, resource constraints

**Monitoring Philosophy:**
- **Proactive**: Detect issues before they impact business
- **Automated**: Scheduled checks with configurable alerts
- **Actionable**: Clear thresholds with remediation procedures
- **Observable**: Centralized dashboards for ML Ops team visibility

---

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources                               │
│  • Production predictions (daily batch)                      │
│  • Ground truth labels (churns, actual CLTV)                │
│  • Feature distributions (Feast feature store)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Monitoring Pipeline (Kedro)                     │
│  src/ai_core/pipelines/monitoring/                           │
│    • Data drift detection (KS test, PSI)                    │
│    • Performance metrics (MAPE, R², calibration)            │
│    • Concept drift detection (target distribution)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              MLflow Tracking                                 │
│  • Log drift metrics per feature                             │
│  • Log model performance over time                           │
│  • Tag runs with alert status                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Alert Manager (Prefect)                         │
│  • Evaluate alert conditions                                 │
│  • Send notifications (Slack, Email, PagerDuty)             │
│  • Create incident tickets (Jira)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Monitoring Dashboard                            │
│  • Streamlit: Real-time model health                         │
│  • MLflow UI: Historical metrics and drift trends           │
│  • Grafana: System-level metrics (latency, throughput)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Drift Detection

### What is Drift?

**Data Drift (Covariate Shift):**
- Changes in input feature distributions (P(X))
- Example: Average customer tenure increases from 6 to 12 months
- **Impact**: Model trained on different distribution may degrade

**Concept Drift (Prior Probability Shift):**
- Changes in target distribution or X → Y relationship (P(Y|X))
- Example: Subscription pricing change affects churn behavior
- **Impact**: Model assumptions violated, predictions inaccurate

### Drift Detection Methods

#### 1. Population Stability Index (PSI)

**What it measures**: Distribution change between reference (training) and production data.

**Formula:**
```
PSI = Σ (% Actual - % Expected) × ln(% Actual / % Expected)
```

**Thresholds:**
- PSI < 0.1: **No significant drift**
- 0.1 ≤ PSI < 0.25: **Moderate drift** (monitor closely)
- PSI ≥ 0.25: **Severe drift** (retrain recommended)

**Configuration** (`conf/base/parameters.yml`):
```yaml
monitoring:
  drift_detection:
    method: "psi"
    psi_thresholds:
      no_drift: 0.1
      moderate_drift: 0.25
    features_to_monitor:
      - frequency
      - recency
      - monetary_value
      - engagement_score
      - watch_time
```

**Implementation**:
```python
# src/ai_core/pipelines/monitoring/nodes.py

import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index for drift detection.

    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (production data)
        bins: Number of bins for discretization

    Returns:
        PSI score (0 = no drift, >0.25 = severe drift)
    """
    # Bin the data
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Calculate proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return psi
```

#### 2. Kolmogorov-Smirnov (KS) Test

**What it measures**: Maximum difference between cumulative distributions.

**Thresholds:**
- p-value > 0.05: **No significant drift** (distributions similar)
- p-value ≤ 0.05: **Drift detected** (distributions differ)

**Implementation**:
```python
from scipy.stats import ks_2samp

def detect_drift_ks_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Detect drift using Kolmogorov-Smirnov test.

    Returns:
        Dict with 'drift_detected' (bool), 'p_value' (float), 'ks_statistic' (float)
    """
    ks_stat, p_value = ks_2samp(reference, current)

    return {
        "drift_detected": p_value <= alpha,
        "p_value": p_value,
        "ks_statistic": ks_stat,
        "severity": "high" if p_value < 0.01 else "moderate" if p_value < 0.05 else "none"
    }
```

#### 3. Jensen-Shannon Divergence

**What it measures**: Symmetric divergence between two probability distributions.

**Thresholds:**
- JSD < 0.1: No drift
- 0.1 ≤ JSD < 0.3: Moderate drift
- JSD ≥ 0.3: Severe drift

```python
from scipy.spatial.distance import jensenshannon

def calculate_js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    """Calculate Jensen-Shannon divergence."""
    # Discretize distributions
    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)

    # Normalize
    hist_p = hist_p / hist_p.sum()
    hist_q = hist_q / hist_q.sum()

    return jensenshannon(hist_p, hist_q)
```

---

## Performance Monitoring

### Key Performance Metrics

#### For Regression (CLTV Prediction)

| Metric | Formula | Target | Alert Threshold |
|--------|---------|--------|-----------------|
| **MAPE** | Mean Absolute Percentage Error | < 18% | > 25% |
| **RMSE** | Root Mean Squared Error | < $50 | > $75 |
| **R²** | Coefficient of Determination | > 0.70 | < 0.60 |
| **MAE** | Mean Absolute Error | < $40 | > $60 |

#### For Classification (Churn Prediction)

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **AUC-ROC** | > 0.85 | < 0.75 |
| **Precision** | > 0.80 | < 0.70 |
| **Recall** | > 0.75 | < 0.65 |
| **F1 Score** | > 0.78 | < 0.68 |

### Monitoring Implementation

```python
# src/ai_core/pipelines/monitoring/nodes.py

import mlflow
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def monitor_model_performance(
    predictions: pl.DataFrame,
    actuals: pl.DataFrame,
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Monitor model performance and detect degradation.

    Args:
        predictions: DataFrame with predicted CLTV values
        actuals: DataFrame with ground truth CLTV (from business data)
        thresholds: Performance thresholds from config

    Returns:
        Dict with metrics and alert status
    """
    # Merge predictions with actuals
    df = predictions.join(actuals, on="customer_id", how="inner")

    # Calculate metrics
    mape = mean_absolute_percentage_error(df["actual_cltv"], df["predicted_cltv"])
    r2 = r2_score(df["actual_cltv"], df["predicted_cltv"])
    mae = (df["actual_cltv"] - df["predicted_cltv"]).abs().mean()

    # Evaluate alerts
    alerts = []
    if mape > thresholds["mape"]:
        alerts.append(f"MAPE degraded: {mape:.2%} > {thresholds['mape']:.2%}")
    if r2 < thresholds["r2"]:
        alerts.append(f"R² degraded: {r2:.3f} < {thresholds['r2']:.3f}")

    # Log to MLflow
    with mlflow.start_run(run_name="performance_monitoring"):
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.set_tag("alert_status", "CRITICAL" if alerts else "OK")

    return {
        "mape": mape,
        "r2": r2,
        "mae": mae,
        "alerts": alerts,
        "status": "degraded" if alerts else "healthy"
    }
```

### Calibration Monitoring

**What is calibration?**
Calibration measures whether predicted probabilities match actual outcomes.

**Example**: If model predicts 70% churn probability for 100 customers, ~70 should actually churn.

```python
from sklearn.calibration import calibration_curve

def monitor_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Monitor prediction calibration for churn model."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error (ECE)
    ece = np.abs(prob_true - prob_pred).mean()

    # Alert if ECE > 0.1 (10% calibration error)
    alert = ece > 0.1

    return {
        "expected_calibration_error": ece,
        "calibration_alert": alert,
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    }
```

---

## Alert Configuration

### Alert Levels

| Level | Severity | Response Time | Notification Channels |
|-------|----------|---------------|----------------------|
| **INFO** | Low | 24 hours | Slack #ml-monitoring |
| **WARNING** | Medium | 4 hours | Slack + Email |
| **CRITICAL** | High | 1 hour | Slack + Email + PagerDuty |
| **EMERGENCY** | Urgent | Immediate | All channels + Phone |

### Alert Rules

```yaml
# conf/base/parameters.yml

monitoring:
  alerts:
    # Data Drift Alerts
    - name: "High PSI Drift"
      condition: "psi >= 0.25"
      level: "WARNING"
      message: "Severe data drift detected on feature {feature_name}. PSI: {psi_value:.3f}"
      action: "Review feature distributions and consider model retraining"

    - name: "Multiple Features Drifting"
      condition: "num_drifted_features >= 3"
      level: "CRITICAL"
      message: "{num_drifted_features} features showing drift. Model inputs significantly changed."
      action: "Initiate emergency model retraining"

    # Performance Alerts
    - name: "MAPE Degradation"
      condition: "mape > 0.25"
      level: "WARNING"
      message: "Model MAPE degraded to {mape:.2%}. Target: <18%."
      action: "Investigate data quality and model assumptions"

    - name: "Critical Performance Drop"
      condition: "mape > 0.35 OR r2 < 0.50"
      level: "CRITICAL"
      message: "Model performance critically degraded. MAPE: {mape:.2%}, R²: {r2:.3f}"
      action: "Immediate model retraining required. Rollback to previous version if available."

    # System Health Alerts
    - name: "Prediction Latency"
      condition: "p95_latency_ms > 1000"
      level: "WARNING"
      message: "Prediction latency degraded: P95 = {p95_latency_ms}ms"
      action: "Check system resources and database query performance"

    - name: "Batch Job Failure"
      condition: "pipeline_status == 'failed'"
      level: "CRITICAL"
      message: "Pipeline {pipeline_name} failed at step {failed_step}"
      action: "Review logs and rerun pipeline"
```

### Notification Integration

#### Slack Alerts

```python
# src/utils/notifications.py

import requests
import os

def send_slack_alert(message: str, level: str = "WARNING"):
    """Send alert to Slack channel."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    color = {
        "INFO": "#36a64f",
        "WARNING": "#ff9900",
        "CRITICAL": "#ff0000",
        "EMERGENCY": "#990000"
    }[level]

    payload = {
        "attachments": [{
            "color": color,
            "title": f"{level}: CLTV Model Alert",
            "text": message,
            "footer": "Predictive CLTV Insights",
            "ts": int(datetime.now().timestamp())
        }]
    }

    response = requests.post(webhook_url, json=payload)
    return response.status_code == 200
```

#### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject: str, body: str, recipients: List[str]):
    """Send alert email to ML Ops team."""
    msg = MIMEText(body)
    msg['Subject'] = f"[CLTV Alert] {subject}"
    msg['From'] = "ml-ops@example.com"
    msg['To'] = ", ".join(recipients)

    with smtplib.SMTP('localhost') as server:
        server.send_message(msg)
```

---

## Monitoring Dashboard

### Streamlit Monitoring Dashboard

Located at: `app/pages/monitoring.py`

**Sections:**
1. **Model Health Overview**
   - Current MAPE, R², MAE
   - Traffic light indicators (🟢 Green / 🟡 Yellow / 🔴 Red)

2. **Drift Detection**
   - PSI trends per feature (last 30 days)
   - Distribution comparison plots (reference vs current)

3. **Performance Trends**
   - MAPE over time (line chart)
   - Prediction error distribution (histogram)

4. **Alert History**
   - Recent alerts table
   - Alert frequency by type

```python
# app/pages/monitoring.py (excerpt)

import streamlit as st
import mlflow
import polars as pl

st.title("🔍 Model Monitoring Dashboard")

# Fetch recent monitoring runs from MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("cltv_monitoring")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=30)

# Extract metrics
mape_history = [run.data.metrics.get("mape") for run in runs]
dates = [run.info.start_time for run in runs]

# Plot MAPE trend
st.line_chart({"Date": dates, "MAPE": mape_history})

# Current status
latest_mape = mape_history[0]
status = "🟢 Healthy" if latest_mape < 0.18 else "🟡 Warning" if latest_mape < 0.25 else "🔴 Critical"
st.metric("Model Status", status, f"MAPE: {latest_mape:.2%}")
```

---

## Model Retraining Triggers

### Automated Retraining Conditions

```yaml
# conf/base/parameters.yml

monitoring:
  retraining_triggers:
    # Scheduled retraining (even if no drift)
    - type: "schedule"
      interval: "monthly"
      description: "Proactive monthly retraining to capture seasonal trends"

    # Performance-based trigger
    - type: "performance_degradation"
      condition: "mape > 0.25 for 3 consecutive days"
      description: "Model accuracy dropped below acceptable threshold"

    # Drift-based trigger
    - type: "data_drift"
      condition: "num_drifted_features >= 3 AND max_psi > 0.3"
      description: "Significant distribution changes detected"

    # Data volume trigger
    - type: "new_data_available"
      condition: "num_new_customers > 10000"
      description: "Sufficient new data accumulated for retraining"
```

### Retraining Workflow

```python
# src/prefect_orchestration/retraining_flow.py

from prefect import flow, task

@task
def check_retraining_conditions(monitoring_results: Dict) -> bool:
    """Evaluate if retraining should be triggered."""
    # Check performance
    if monitoring_results["mape"] > 0.25:
        return True

    # Check drift
    if monitoring_results["num_drifted_features"] >= 3:
        return True

    # Check schedule
    last_training = get_last_training_date()
    if (datetime.now() - last_training).days > 30:
        return True

    return False

@flow(name="adaptive_retraining")
def adaptive_retraining_flow():
    """Conditional model retraining based on monitoring results."""
    # Run monitoring pipeline
    monitoring_results = run_monitoring_pipeline()

    # Evaluate conditions
    should_retrain = check_retraining_conditions(monitoring_results)

    if should_retrain:
        send_slack_alert("🔄 Initiating model retraining...", level="INFO")

        # Run training pipelines
        run_data_processing_pipeline()
        run_data_science_pipeline()
        run_evaluation_pipeline()

        send_slack_alert("✅ Model retraining completed successfully", level="INFO")
    else:
        send_slack_alert("✅ Model health check passed. No retraining needed.", level="INFO")
```

---

## Incident Response

### Response Procedures by Alert Level

#### WARNING Level

1. **Acknowledge alert** (within 4 hours)
2. **Investigate root cause**:
   - Check MLflow dashboard for metric trends
   - Review recent data changes
   - Compare feature distributions
3. **Document findings** in incident log
4. **Schedule retraining** if drift confirmed
5. **Update alert thresholds** if false positive

#### CRITICAL Level

1. **Immediate acknowledgment** (within 1 hour)
2. **Assess business impact**:
   - Are predictions still being served?
   - What % of customers affected?
   - Revenue impact estimation
3. **Mitigation options**:
   - **Option A**: Rollback to previous model version (fastest)
   - **Option B**: Emergency retraining (4-6 hours)
   - **Option C**: Manual overrides for high-value customers
4. **Execute mitigation plan**
5. **Post-incident review** within 48 hours

### Rollback Procedure

```bash
# Rollback to previous model version

# 1. Identify last known good version
mlflow models list --model-name cltv_production

# 2. Promote previous version to production
mlflow models update \
  --model-name cltv_production \
  --version 42 \
  --stage Production

# 3. Restart prediction service
systemctl restart cltv-api

# 4. Verify rollback
curl -X POST http://localhost:8000/v1/predict \
  -H "X-API-Key: $API_KEY" \
  -d '{"entity_id": "test_customer", "features": {...}}'
```

---

## Troubleshooting

### Issue 1: False Positive Drift Alerts

**Symptom**: PSI alerts triggering but predictions still accurate.

**Diagnosis**:
- Check if drift is in non-critical features
- Verify ground truth data quality
- Review PSI threshold appropriateness

**Solution**:
```yaml
# Adjust PSI thresholds or exclude noisy features
monitoring:
  drift_detection:
    psi_thresholds:
      moderate_drift: 0.30  # Increased from 0.25
    features_to_exclude:
      - device_type  # High variance, low predictive power
```

### Issue 2: Monitoring Pipeline Fails

**Symptom**: Monitoring pipeline throws error, no metrics logged.

**Diagnosis**:
```bash
# Check Prefect logs
prefect deployment run monitoring-pipeline/production --watch

# Check MLflow tracking server
curl http://localhost:5000/health
```

**Solution**:
- Verify MLflow tracking server is running
- Check feature store data availability
- Ensure ground truth labels are populated

### Issue 3: High Latency in Monitoring

**Symptom**: Monitoring pipeline takes > 30 minutes to run.

**Solution**:
- Implement sampling for large datasets
- Use incremental drift detection (last N days only)
- Parallelize feature-level drift calculations

---

## Additional Resources

- [Evidently AI Drift Detection](https://docs.evidentlyai.com/)
- [MLflow Model Monitoring](https://mlflow.org/docs/latest/model-monitoring.html)
- [Google - Rules of ML: Monitoring](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_5_test_the_infrastructure_independently_from_the_machine_learning)
- Project-specific: `docs/runbook.md` (Incident response procedures)
