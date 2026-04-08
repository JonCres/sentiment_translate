# Deployment Guide

**Project:** Predictive CLTV Insights
**Industry:** Media & Entertainment
**Last Updated:** 2026-02-13

---

## Table of Contents

1. [Overview](#overview)
2. [Deployment Architecture](#deployment-architecture)
3. [Prerequisites](#prerequisites)
4. [Local Deployment](#local-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Cloud Deployments](#cloud-deployments)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Secrets Management](#secrets-management)
10. [Monitoring & Observability](#monitoring--observability)
11. [Scaling](#scaling)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying the Predictive CLTV Insights system from development to production across multiple environments and platforms.

**Deployment Modes:**
- **Development**: Local laptop (uv + Python 3.12)
- **Staging**: Docker Compose (isolated testing)
- **Production**: Kubernetes (AWS EKS, GCP GKE, or Azure AKS)

**Components to Deploy:**
1. **Kedro Pipelines** (data processing, model training)
2. **Prefect Orchestration** (scheduled workflows)
3. **MLflow Tracking Server** (experiment tracking)
4. **Feast Feature Store** (feature serving)
5. **Streamlit Dashboard** (business UI)
6. **API Service** (REST predictions) *(optional)*

---

## Deployment Architecture

### Production Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Load Balancer (ALB/GCP LB)              │
└──────────────────┬─────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌───────────────┐      ┌──────────────┐
│   Streamlit   │      │   API Service│
│   Dashboard   │      │   (FastAPI)  │
│   (Public)    │      │   (Private)  │
└───────┬───────┘      └──────┬───────┘
        │                     │
        └─────────┬───────────┘
                  ↓
        ┌─────────────────────┐
        │   Kubernetes Cluster │
        │  ┌─────────────────┐│
        │  │ Prefect Workers ││  ← Scheduled jobs
        │  │ (Kedro Runners) ││
        │  └─────────────────┘│
        │  ┌─────────────────┐│
        │  │ MLflow Server   ││  ← Experiment tracking
        │  └─────────────────┘│
        │  ┌─────────────────┐│
        │  │ Feast Online    ││  ← Feature serving
        │  │ (Redis)         ││
        │  └─────────────────┘│
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌───────────────┐      ┌──────────────┐
│   PostgreSQL  │      │   S3/GCS     │
│   (Metadata)  │      │   (Artifacts)│
└───────────────┘      └──────────────┘
```

---

## Prerequisites

### Infrastructure Requirements

**Minimum Resources (Development):**
- 4 CPU cores
- 16 GB RAM
- 50 GB disk space

**Recommended Resources (Production):**
- 16+ CPU cores (or 8 with GPU/XPU)
- 64 GB RAM
- 500 GB SSD storage
- GPU/XPU for model training (optional but recommended)

### Software Requirements

- **Container Runtime**: Docker 24+ or containerd
- **Orchestrator**: Kubernetes 1.28+
- **CI/CD**: GitHub Actions, GitLab CI, or Jenkins
- **Cloud CLI**: `aws`, `gcloud`, or `az` (depending on cloud provider)
- **Kubernetes CLI**: `kubectl` and `helm` 3.0+

---

## Local Deployment

For development and testing on your local machine.

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Wizeline/ai_cores.git
cd ai_cores/cores/media_entertainment/predictive-CLTV-insights

# Install dependencies
uv sync --extra mps  # or cuda/xpu

# Set environment variables
export MLFLOW_TRACKING_URI="http://localhost:5000"
export GROQ_API_KEY="gsk_..."  # Optional: for LLM interpretations
```

### 2. Start MLflow Server

```bash
# Terminal 1: MLflow tracking server
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlruns
```

### 3. Run Pipelines

```bash
# Terminal 2: Run Kedro pipelines
uv run kedro run --pipeline=data_processing
uv run kedro run --pipeline=data_science
uv run kedro run --pipeline=visualization

# Or run via Prefect orchestration
uv run python src/prefect_orchestration/run_all_pipelines.py
```

### 4. Start Dashboard

```bash
# Terminal 3: Streamlit dashboard
uv run streamlit run app/app.py
```

Access dashboard at: `http://localhost:8501`

---

## Docker Deployment

Containerized deployment using Docker and Docker Compose.

### 1. Build Docker Images

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy application code
COPY . .

# Expose ports
EXPOSE 8501 8000

# Default command (override in docker-compose)
CMD ["uv", "run", "streamlit", "run", "app/app.py"]
```

Build image:
```bash
docker build -t cltv-insights:latest .
```

### 2. Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL for metadata
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # MLflow Tracking Server
  mlflow:
    image: cltv-insights:latest
    command: >
      uv run mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlflow:${POSTGRES_PASSWORD}@postgres/mlflow
      --default-artifact-root s3://cltv-artifacts/
    environment:
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    ports:
      - "5000:5000"
    depends_on:
      - postgres

  # Redis for Feast online store
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Prefect Worker (Pipeline Orchestration)
  prefect-worker:
    image: cltv-insights:latest
    command: uv run prefect worker start --pool default-pool
    environment:
      PREFECT_API_URL: ${PREFECT_API_URL}
      MLFLOW_TRACKING_URI: http://mlflow:5000
      GROQ_API_KEY: ${GROQ_API_KEY}
    volumes:
      - ./data:/app/data
      - ./mlruns:/app/mlruns
    depends_on:
      - mlflow
      - redis

  # Streamlit Dashboard
  dashboard:
    image: cltv-insights:latest
    command: uv run streamlit run app/app.py
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8501:8501"
    depends_on:
      - mlflow

volumes:
  postgres_data:
  redis_data:
```

### 3. Deploy with Docker Compose

```bash
# Set environment variables
export POSTGRES_PASSWORD="secure_password"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/..."
export GROQ_API_KEY="gsk_..."

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

Production-grade deployment on Kubernetes.

### 1. Namespace Setup

```bash
# Create namespace
kubectl create namespace cltv-production

# Set as default namespace
kubectl config set-context --current --namespace=cltv-production
```

### 2. Secrets Management

Create Kubernetes secrets:

```bash
# MLflow database credentials
kubectl create secret generic mlflow-db \
  --from-literal=username=mlflow \
  --from-literal=password=YOUR_PASSWORD

# AWS credentials for S3
kubectl create secret generic aws-credentials \
  --from-literal=access_key_id=YOUR_KEY \
  --from-literal=secret_access_key=YOUR_SECRET

# API keys
kubectl create secret generic api-keys \
  --from-literal=groq_api_key=gsk_...
```

### 3. ConfigMap

Create `configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cltv-config
  namespace: cltv-production
data:
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  FEAST_ONLINE_STORE: "redis://redis-service:6379"
  LOG_LEVEL: "INFO"
```

Apply:
```bash
kubectl apply -f configmap.yaml
```

### 4. Deployment Manifests

**MLflow Deployment** (`mlflow-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: cltv-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: cltv-insights:latest
        command: ["uv", "run", "mlflow", "server"]
        args:
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "5000"
          - "--backend-store-uri"
          - "postgresql://$(DB_USER):$(DB_PASSWORD)@postgres-service/mlflow"
          - "--default-artifact-root"
          - "s3://cltv-artifacts/"
        env:
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: mlflow-db
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db
              key: password
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access_key_id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret_access_key
        ports:
        - containerPort: 5000
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: cltv-production
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
```

**Prefect Worker Deployment** (`prefect-worker-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prefect-worker
  namespace: cltv-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prefect-worker
  template:
    metadata:
      labels:
        app: prefect-worker
    spec:
      containers:
      - name: worker
        image: cltv-insights:latest
        command: ["uv", "run", "prefect", "worker", "start"]
        args:
          - "--pool"
          - "production-pool"
        env:
        - name: PREFECT_API_URL
          value: "https://api.prefect.cloud/api/accounts/YOUR_ACCOUNT/workspaces/YOUR_WORKSPACE"
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: cltv-config
              key: MLFLOW_TRACKING_URI
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: groq_api_key
        volumeMounts:
        - name: data
          mountPath: /app/data
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
          limits:
            cpu: "8"
            memory: "32Gi"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: cltv-data-pvc
```

**Streamlit Dashboard** (`dashboard-deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard
  namespace: cltv-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dashboard
  template:
    metadata:
      labels:
        app: dashboard
    spec:
      containers:
      - name: streamlit
        image: cltv-insights:latest
        command: ["uv", "run", "streamlit", "run", "app/app.py"]
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: cltv-config
              key: MLFLOW_TRACKING_URI
        ports:
        - containerPort: 8501
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "1"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
  namespace: cltv-production
spec:
  selector:
    app: dashboard
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### 5. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f mlflow-deployment.yaml
kubectl apply -f prefect-worker-deployment.yaml
kubectl apply -f dashboard-deployment.yaml

# Check deployment status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/mlflow
kubectl logs -f deployment/prefect-worker
```

---

## Cloud Deployments

### AWS (EKS)

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name cltv-production \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# 2. Configure kubectl
aws eks update-kubeconfig --name cltv-production --region us-west-2

# 3. Deploy (follow Kubernetes section above)
```

### GCP (GKE)

```bash
# 1. Create GKE cluster
gcloud container clusters create cltv-production \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5

# 2. Get credentials
gcloud container clusters get-credentials cltv-production --zone us-central1-a

# 3. Deploy (follow Kubernetes section above)
```

### Azure (AKS)

```bash
# 1. Create resource group
az group create --name cltv-rg --location eastus

# 2. Create AKS cluster
az aks create \
  --resource-group cltv-rg \
  --name cltv-production \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 5

# 3. Get credentials
az aks get-credentials --resource-group cltv-rg --name cltv-production

# 4. Deploy (follow Kubernetes section above)
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy CLTV Insights

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest tests/ --cov=src

      - name: Lint
        run: |
          uv run black --check src/
          uv run flake8 src/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t cltv-insights:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker tag cltv-insights:${{ github.sha }} myregistry/cltv-insights:latest
          docker push myregistry/cltv-insights:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG }}" > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/mlflow mlflow=myregistry/cltv-insights:latest
          kubectl set image deployment/prefect-worker worker=myregistry/cltv-insights:latest
          kubectl set image deployment/dashboard streamlit=myregistry/cltv-insights:latest
          kubectl rollout status deployment/mlflow
```

---

## Secrets Management

### Using AWS Secrets Manager

```python
# src/utils/secrets.py

import boto3
import json

def get_secret(secret_name: str, region: str = "us-west-2") -> dict:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name=region)

    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage in nodes
credentials = get_secret("cltv-production/credentials")
mlflow_uri = credentials["mlflow_tracking_uri"]
```

### Using Kubernetes Secrets

Secrets are automatically mounted as environment variables (see Kubernetes section).

---

## Monitoring & Observability

### Prometheus + Grafana

Deploy monitoring stack:

```bash
# Install Prometheus operator
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

### MLflow Monitoring

MLflow UI provides built-in monitoring at `http://mlflow-service:5000`

### Custom Metrics

Expose Prefect metrics:
```python
# src/prefect_orchestration/monitoring.py

from prefect import flow
from prometheus_client import Counter, Histogram

pipeline_runs = Counter('pipeline_runs_total', 'Total pipeline runs')
pipeline_duration = Histogram('pipeline_duration_seconds', 'Pipeline duration')

@flow
def monitored_pipeline():
    pipeline_runs.inc()
    with pipeline_duration.time():
        # Pipeline logic
        ...
```

---

## Scaling

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prefect-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prefect-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

Increase resources for model training:
```yaml
resources:
  requests:
    cpu: "8"
    memory: "32Gi"
    nvidia.com/gpu: "1"  # Request GPU
  limits:
    cpu: "16"
    memory: "64Gi"
    nvidia.com/gpu: "1"
```

---

## Troubleshooting

### Issue: Pods failing with OOMKilled

**Solution**: Increase memory limits or optimize model training batch size.

### Issue: MLflow artifacts not persisting

**Solution**: Verify S3/GCS permissions and bucket configuration.

### Issue: Prefect workers not picking up jobs

**Solution**: Check `PREFECT_API_URL` and work pool configuration.

---

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Prefect Deployment Guide](https://docs.prefect.io/latest/guides/deployment/)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployment/index.html)
