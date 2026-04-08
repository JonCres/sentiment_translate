# API Specification

**AI Core:** `Predictive CLTV Insights`  
**Industry:** `Media & Entertainment`  
**API Version:** `v1.0.0`  
**OpenAPI Version:** `3.0.3`  
**Last Updated:** `2026-01-21`

---

## OpenAPI Specification

```yaml
openapi: 3.0.3
info:
  title: "Predictive CLTV Insights API"
  description: |
    RESTful API for the Predictive CLTV Insights AI Core.
    
    **Capabilities:**
    - Prediction inference endpoint
    - Model metadata retrieval
    - Feature importance explanation
    - Batch prediction submission
    
    **Authentication:** API Key (Header: `X-API-Key`)
    
    **Rate Limits:** 
    - Standard: 100 requests/minute
    - Batch: 10 requests/minute
  version: "1.0.0"
  contact:
    name: "AI Core Team"
    email: "team@aicore.example.com"
  license:
    name: "Proprietary"

servers:
  - url: "https://api.example.com/v1"
    description: "Production server"
  - url: "https://staging-api.example.com/v1"
    description: "Staging server"
  - url: "http://localhost:8000/v1"
    description: "Local development server"

security:
  - ApiKeyAuth: []

tags:
  - name: "Predictions"
    description: "Model inference endpoints"
  - name: "Models"
    description: "Model metadata and management"
  - name: "Explainability"
    description: "Model interpretation and feature importance"
  - name: "Health"
    description: "Service health and status"

paths:
  /health:
    get:
      tags:
        - "Health"
      summary: "Health check endpoint"
      description: "Returns the health status of the API service"
      operationId: "getHealth"
      security: []
      responses:
        "200":
          description: "Service is healthy"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
              example:
                status: "healthy"
                version: "1.0.0"
                timestamp: "2026-01-21T20:07:00Z"
                dependencies:
                  mlflow: "healthy"
                  feast: "healthy"
                  database: "healthy"

  /predict:
    post:
      tags:
        - "Predictions"
      summary: "Single prediction inference"
      description: |
        Generate a prediction for a single entity (e.g., customer, subscriber).
        
        **Example Use Cases:**
        - Predict churn probability for a subscriber
        - Forecast CLTV for a customer
        - Estimate risk score for an account
      operationId: "predictSingle"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/PredictionRequest"
            example:
              entity_id: "user_12345"
              features:
                tenure_days: 365
                avg_session_duration: 45.2
                monthly_spend: 29.99
                device_type: "mobile"
                engagement_score: 0.78
              model_version: "production"
      responses:
        "200":
          description: "Prediction generated successfully"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/PredictionResponse"
              example:
                entity_id: "user_12345"
                prediction:
                  score: 0.73
                  label: "high_risk"
                  confidence_interval:
                    lower: 0.68
                    upper: 0.78
                  percentile_rank: 85
                explanation:
                  top_features:
                    - feature: "engagement_score"
                      importance: 0.42
                      direction: "negative"
                    - feature: "tenure_days"
                      importance: 0.28
                      direction: "positive"
                    - feature: "monthly_spend"
                      importance: 0.18
                      direction: "positive"
                metadata:
                  model_version: "production_v1.2.3"
                  prediction_timestamp: "2026-01-21T20:07:00Z"
                  inference_time_ms: 45
        "400":
          $ref: "#/components/responses/BadRequest"
        "401":
          $ref: "#/components/responses/Unauthorized"
        "429":
          $ref: "#/components/responses/RateLimitExceeded"
        "500":
          $ref: "#/components/responses/InternalServerError"

  /predict/batch:
    post:
      tags:
        - "Predictions"
      summary: "Batch prediction inference"
      description: |
        Generate predictions for multiple entities in a single request.
        
        **Limits:**
        - Maximum 1000 entities per batch
        - Asynchronous processing for batches > 100 entities
      operationId: "predictBatch"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/BatchPredictionRequest"
            example:
              entities:
                - entity_id: "user_12345"
                  features:
                    tenure_days: 365
                    engagement_score: 0.78
                - entity_id: "user_67890"
                  features:
                    tenure_days: 120
                    engagement_score: 0.45
              model_version: "production"
              async: true
      responses:
        "202":
          description: "Batch prediction job accepted (async)"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BatchJobResponse"
              example:
                job_id: "batch_abc123"
                status: "processing"
                total_entities: 2
                estimated_completion: "2026-01-21T20:10:00Z"
                status_url: "/predict/batch/batch_abc123/status"
        "200":
          description: "Batch predictions completed (sync)"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BatchPredictionResponse"
        "400":
          $ref: "#/components/responses/BadRequest"
        "401":
          $ref: "#/components/responses/Unauthorized"

  /predict/batch/{job_id}/status:
    get:
      tags:
        - "Predictions"
      summary: "Get batch job status"
      description: "Retrieve the status and results of a batch prediction job"
      operationId: "getBatchJobStatus"
      parameters:
        - name: "job_id"
          in: "path"
          required: true
          schema:
            type: "string"
          example: "batch_abc123"
      responses:
        "200":
          description: "Job status retrieved successfully"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BatchJobStatusResponse"
              example:
                job_id: "batch_abc123"
                status: "completed"
                progress:
                  total: 2
                  completed: 2
                  failed: 0
                results_url: "/predict/batch/batch_abc123/results"
                created_at: "2026-01-21T20:07:00Z"
                completed_at: "2026-01-21T20:09:30Z"
        "404":
          $ref: "#/components/responses/NotFound"

  /models:
    get:
      tags:
        - "Models"
      summary: "List available models"
      description: "Retrieve metadata for all registered models"
      operationId: "listModels"
      responses:
        "200":
          description: "Models retrieved successfully"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ModelsListResponse"
              example:
                models:
                  - model_id: "production_v1.2.3"
                    model_type: "xgboost_classifier"
                    status: "active"
                    metrics:
                      auc_roc: 0.94
                      precision: 0.87
                      recall: 0.89
                    created_at: "2026-01-15T10:00:00Z"
                    is_default: true
                  - model_id: "champion_v1.2.2"
                    model_type: "xgboost_classifier"
                    status: "archived"
                    metrics:
                      auc_roc: 0.92
                      precision: 0.85
                      recall: 0.88
                    created_at: "2026-01-01T10:00:00Z"
                    is_default: false

  /models/{model_id}:
    get:
      tags:
        - "Models"
      summary: "Get model metadata"
      description: "Retrieve detailed metadata for a specific model"
      operationId: "getModel"
      parameters:
        - name: "model_id"
          in: "path"
          required: true
          schema:
            type: "string"
          example: "production_v1.2.3"
      responses:
        "200":
          description: "Model metadata retrieved successfully"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ModelMetadataResponse"
              example:
                model_id: "production_v1.2.3"
                model_type: "xgboost_classifier"
                framework: "xgboost"
                framework_version: "2.0.3"
                status: "active"
                metrics:
                  auc_roc: 0.94
                  precision: 0.87
                  recall: 0.89
                  f1_score: 0.88
                  c_index: 0.91
                hyperparameters:
                  max_depth: 6
                  learning_rate: 0.1
                  n_estimators: 100
                feature_importance:
                  - feature: "engagement_score"
                    importance: 0.42
                  - feature: "tenure_days"
                    importance: 0.28
                training_data:
                  dataset_version: "2026-01-10"
                  num_samples: 1000000
                  num_features: 45
                created_at: "2026-01-15T10:00:00Z"
                created_by: "ml_pipeline_v2"
        "404":
          $ref: "#/components/responses/NotFound"

  /explain:
    post:
      tags:
        - "Explainability"
      summary: "Generate prediction explanation"
      description: |
        Generate SHAP-based or LIME-based explanation for a prediction.
        
        **Methods:**
        - `shap`: SHapley Additive exPlanations (global/local)
        - `lime`: Local Interpretable Model-agnostic Explanations
      operationId: "explainPrediction"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ExplainRequest"
            example:
              entity_id: "user_12345"
              features:
                tenure_days: 365
                engagement_score: 0.78
              method: "shap"
              model_version: "production"
      responses:
        "200":
          description: "Explanation generated successfully"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ExplainResponse"
              example:
                entity_id: "user_12345"
                prediction_score: 0.73
                explanation:
                  method: "shap"
                  feature_contributions:
                    - feature: "engagement_score"
                      shap_value: -0.15
                      feature_value: 0.78
                      impact: "negative"
                    - feature: "tenure_days"
                      shap_value: 0.08
                      feature_value: 365
                      impact: "positive"
                  base_value: 0.50
                  visualization_url: "/explain/user_12345/waterfall.png"

components:
  securitySchemes:
    ApiKeyAuth:
      type: "apiKey"
      in: "header"
      name: "X-API-Key"

  schemas:
    HealthResponse:
      type: "object"
      properties:
        status:
          type: "string"
          enum: ["healthy", "degraded", "unhealthy"]
        version:
          type: "string"
        timestamp:
          type: "string"
          format: "date-time"
        dependencies:
          type: "object"
          additionalProperties:
            type: "string"

    PredictionRequest:
      type: "object"
      required:
        - "entity_id"
        - "features"
      properties:
        entity_id:
          type: "string"
          description: "Unique identifier for the entity"
        features:
          type: "object"
          description: "Feature values for prediction"
          additionalProperties: true
        model_version:
          type: "string"
          description: "Model version to use (default: production)"
          default: "production"

    PredictionResponse:
      type: "object"
      properties:
        entity_id:
          type: "string"
        prediction:
          type: "object"
          properties:
            score:
              type: "number"
              format: "float"
            label:
              type: "string"
            confidence_interval:
              type: "object"
              properties:
                lower:
                  type: "number"
                upper:
                  type: "number"
            percentile_rank:
              type: "integer"
        explanation:
          type: "object"
          properties:
            top_features:
              type: "array"
              items:
                type: "object"
                properties:
                  feature:
                    type: "string"
                  importance:
                    type: "number"
                  direction:
                    type: "string"
                    enum: ["positive", "negative"]
        metadata:
          type: "object"
          properties:
            model_version:
              type: "string"
            prediction_timestamp:
              type: "string"
              format: "date-time"
            inference_time_ms:
              type: "integer"

    BatchPredictionRequest:
      type: "object"
      required:
        - "entities"
      properties:
        entities:
          type: "array"
          items:
            $ref: "#/components/schemas/PredictionRequest"
          maxItems: 1000
        model_version:
          type: "string"
          default: "production"
        async:
          type: "boolean"
          default: true
          description: "Process asynchronously (recommended for >100 entities)"

    BatchPredictionResponse:
      type: "object"
      properties:
        predictions:
          type: "array"
          items:
            $ref: "#/components/schemas/PredictionResponse"
        metadata:
          type: "object"
          properties:
            total_entities:
              type: "integer"
            successful:
              type: "integer"
            failed:
              type: "integer"
            processing_time_ms:
              type: "integer"

    BatchJobResponse:
      type: "object"
      properties:
        job_id:
          type: "string"
        status:
          type: "string"
          enum: ["queued", "processing", "completed", "failed"]
        total_entities:
          type: "integer"
        estimated_completion:
          type: "string"
          format: "date-time"
        status_url:
          type: "string"

    BatchJobStatusResponse:
      type: "object"
      properties:
        job_id:
          type: "string"
        status:
          type: "string"
          enum: ["queued", "processing", "completed", "failed"]
        progress:
          type: "object"
          properties:
            total:
              type: "integer"
            completed:
              type: "integer"
            failed:
              type: "integer"
        results_url:
          type: "string"
        created_at:
          type: "string"
          format: "date-time"
        completed_at:
          type: "string"
          format: "date-time"

    ModelsListResponse:
      type: "object"
      properties:
        models:
          type: "array"
          items:
            type: "object"
            properties:
              model_id:
                type: "string"
              model_type:
                type: "string"
              status:
                type: "string"
                enum: ["active", "archived", "deprecated"]
              metrics:
                type: "object"
                additionalProperties:
                  type: "number"
              created_at:
                type: "string"
                format: "date-time"
              is_default:
                type: "boolean"

    ModelMetadataResponse:
      type: "object"
      properties:
        model_id:
          type: "string"
        model_type:
          type: "string"
        framework:
          type: "string"
        framework_version:
          type: "string"
        status:
          type: "string"
        metrics:
          type: "object"
          additionalProperties:
            type: "number"
        hyperparameters:
          type: "object"
          additionalProperties: true
        feature_importance:
          type: "array"
          items:
            type: "object"
            properties:
              feature:
                type: "string"
              importance:
                type: "number"
        training_data:
          type: "object"
          properties:
            dataset_version:
              type: "string"
            num_samples:
              type: "integer"
            num_features:
              type: "integer"
        created_at:
          type: "string"
          format: "date-time"
        created_by:
          type: "string"

    ExplainRequest:
      type: "object"
      required:
        - "entity_id"
        - "features"
      properties:
        entity_id:
          type: "string"
        features:
          type: "object"
          additionalProperties: true
        method:
          type: "string"
          enum: ["shap", "lime"]
          default: "shap"
        model_version:
          type: "string"
          default: "production"

    ExplainResponse:
      type: "object"
      properties:
        entity_id:
          type: "string"
        prediction_score:
          type: "number"
        explanation:
          type: "object"
          properties:
            method:
              type: "string"
            feature_contributions:
              type: "array"
              items:
                type: "object"
                properties:
                  feature:
                    type: "string"
                  shap_value:
                    type: "number"
                  feature_value:
                    type: "number"
                  impact:
                    type: "string"
                    enum: ["positive", "negative"]
            base_value:
              type: "number"
            visualization_url:
              type: "string"

    ErrorResponse:
      type: "object"
      properties:
        error:
          type: "object"
          properties:
            code:
              type: "string"
            message:
              type: "string"
            details:
              type: "object"
              additionalProperties: true
        timestamp:
          type: "string"
          format: "date-time"
        request_id:
          type: "string"

  responses:
    BadRequest:
      description: "Invalid request parameters"
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ErrorResponse"
          example:
            error:
              code: "INVALID_REQUEST"
              message: "Missing required field: entity_id"
            timestamp: "2026-01-21T20:07:00Z"
            request_id: "req_xyz789"

    Unauthorized:
      description: "Authentication required or invalid API key"
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ErrorResponse"
          example:
            error:
              code: "UNAUTHORIZED"
              message: "Invalid or missing API key"
            timestamp: "2026-01-21T20:07:00Z"
            request_id: "req_xyz789"

    NotFound:
      description: "Resource not found"
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ErrorResponse"
          example:
            error:
              code: "NOT_FOUND"
              message: "Model not found: invalid_model_id"
            timestamp: "2026-01-21T20:07:00Z"
            request_id: "req_xyz789"

    RateLimitExceeded:
      description: "Rate limit exceeded"
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ErrorResponse"
          example:
            error:
              code: "RATE_LIMIT_EXCEEDED"
              message: "Rate limit exceeded: 100 requests per minute"
              details:
                limit: 100
                window: "1 minute"
                retry_after: 45
            timestamp: "2026-01-21T20:07:00Z"
            request_id: "req_xyz789"

    InternalServerError:
      description: "Internal server error"
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ErrorResponse"
          example:
            error:
              code: "INTERNAL_ERROR"
              message: "An unexpected error occurred"
            timestamp: "2026-01-21T20:07:00Z"
            request_id: "req_xyz789"
```

---

## Quick Reference

### Authentication

**Obtaining API Keys:**
1. Navigate to AI Core dashboard: `https://dashboard.example.com/api-keys`
2. Click "Generate New API Key"
3. Assign permissions: `predict:read`, `models:read`, `explain:read`
4. Copy key immediately (shown once): `aik_live_1234567890abcdef...`
5. Store securely (environment variable recommended)

**Key Management:**
- Keys are 64-character strings prefixed with `aik_live_` (production) or `aik_test_` (development)
- Rotate keys every 90 days (automated expiration reminders sent)
- Revoke compromised keys immediately via dashboard
- Scope keys to minimum required permissions (principle of least privilege)

**Using API Keys:**

cURL:
```bash
curl -H "X-API-Key: your_api_key_here" https://api.example.com/v1/health
```

Python:
```python
import requests

headers = {"X-API-Key": "your_api_key_here"}
response = requests.get("https://api.example.com/v1/health", headers=headers)
print(response.json())
```

Environment Variable (Recommended):
```bash
export CLTV_API_KEY="aik_live_1234567890abcdef..."
```
```python
import os
import requests

api_key = os.environ["CLTV_API_KEY"]
headers = {"X-API-Key": api_key}
```

### Single Prediction

**cURL:**
```bash
curl -X POST https://api.example.com/v1/predict \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "user_12345",
    "features": {
      "tenure_days": 365,
      "engagement_score": 0.78,
      "monthly_spend": 29.99
    }
  }'
```

**Python:**
```python
import requests
import os

API_URL = "https://api.example.com/v1"
headers = {"X-API-Key": os.environ["CLTV_API_KEY"]}

# Single customer CLTV prediction
payload = {
    "entity_id": "user_12345",
    "features": {
        "tenure_days": 365,
        "engagement_score": 0.78,
        "monthly_spend": 29.99,
        "device_type": "mobile"
    },
    "model_version": "production"
}

response = requests.post(
    f"{API_URL}/predict",
    headers=headers,
    json=payload
)

if response.status_code == 200:
    result = response.json()
    print(f"Customer: {result['entity_id']}")
    print(f"Churn Risk Score: {result['prediction']['score']:.2f}")
    print(f"Risk Label: {result['prediction']['label']}")
    print(f"Confidence Interval: [{result['prediction']['confidence_interval']['lower']:.2f}, "
          f"{result['prediction']['confidence_interval']['upper']:.2f}]")
    print(f"\nTop Risk Factors:")
    for feature in result['explanation']['top_features']:
        print(f"  - {feature['feature']}: {feature['importance']:.2f} ({feature['direction']})")
else:
    print(f"Error: {response.status_code} - {response.json()}")
```

**Expected Response:**
```json
{
  "entity_id": "user_12345",
  "prediction": {
    "score": 0.73,
    "label": "high_risk",
    "confidence_interval": {"lower": 0.68, "upper": 0.78},
    "percentile_rank": 85
  },
  "explanation": {
    "top_features": [
      {"feature": "engagement_score", "importance": 0.42, "direction": "negative"},
      {"feature": "tenure_days", "importance": 0.28, "direction": "positive"}
    ]
  },
  "metadata": {
    "model_version": "production_v1.2.3",
    "prediction_timestamp": "2026-01-21T20:07:00Z",
    "inference_time_ms": 45
  }
}
```

### Batch Prediction

**cURL:**
```bash
curl -X POST https://api.example.com/v1/predict/batch \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: "application/json" \
  -d '{
    "entities": [
      {"entity_id": "user_1", "features": {"tenure_days": 365, "engagement_score": 0.78}},
      {"entity_id": "user_2", "features": {"tenure_days": 120, "engagement_score": 0.45}}
    ],
    "async": true
  }'
```

**Python (Asynchronous Batch):**
```python
import requests
import time
import os

API_URL = "https://api.example.com/v1"
headers = {"X-API-Key": os.environ["CLTV_API_KEY"]}

# Submit batch job (recommended for > 100 customers)
batch_payload = {
    "entities": [
        {"entity_id": f"user_{i}", "features": {"tenure_days": 365, "engagement_score": 0.78}}
        for i in range(1000)  # Batch of 1000 customers
    ],
    "model_version": "production",
    "async": True
}

# Submit job
response = requests.post(f"{API_URL}/predict/batch", headers=headers, json=batch_payload)
job = response.json()
job_id = job["job_id"]
print(f"Batch job submitted: {job_id}")
print(f"Status URL: {job['status_url']}")
print(f"Estimated completion: {job['estimated_completion']}")

# Poll for completion
while True:
    status_response = requests.get(f"{API_URL}/predict/batch/{job_id}/status", headers=headers)
    status = status_response.json()

    if status["status"] == "completed":
        print(f"\nBatch completed!")
        print(f"Total: {status['progress']['total']}, Successful: {status['progress']['completed']}")

        # Fetch results
        results_response = requests.get(status["results_url"], headers=headers)
        results = results_response.json()

        # Process results
        high_risk_customers = [
            p for p in results["predictions"]
            if p["prediction"]["label"] == "high_risk"
        ]
        print(f"High-risk customers: {len(high_risk_customers)}")
        break
    elif status["status"] == "failed":
        print(f"Batch failed: {status}")
        break
    else:
        print(f"Status: {status['status']}, Progress: {status['progress']['completed']}/{status['progress']['total']}")
        time.sleep(5)  # Poll every 5 seconds
```

**Python (Synchronous Small Batch):**
```python
# For small batches (<100 customers), use synchronous mode
small_batch = {
    "entities": [
        {"entity_id": "user_1", "features": {"tenure_days": 365, "engagement_score": 0.78}},
        {"entity_id": "user_2", "features": {"tenure_days": 120, "engagement_score": 0.45}}
    ],
    "model_version": "production",
    "async": False  # Synchronous
}

response = requests.post(f"{API_URL}/predict/batch", headers=headers, json=small_batch)
results = response.json()

# Immediate results
for pred in results["predictions"]:
    print(f"{pred['entity_id']}: Risk={pred['prediction']['score']:.2f}, Label={pred['prediction']['label']}")
```

---

## Error Handling Best Practices

### Retry Strategy

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

def create_api_session():
    """Create requests session with automatic retries for transient errors."""
    session = requests.Session()

    # Retry on 429 (rate limit), 500, 502, 503, 504 (server errors)
    retry_strategy = Retry(
        total=3,  # Max retries
        backoff_factor=2,  # Wait 2^n seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

# Usage
session = create_api_session()
response = session.post(
    "https://api.example.com/v1/predict",
    headers={"X-API-Key": os.environ["CLTV_API_KEY"]},
    json=payload
)
```

### Error Response Handling

```python
def make_prediction(entity_id, features):
    """Make prediction with comprehensive error handling."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"X-API-Key": os.environ["CLTV_API_KEY"]},
            json={"entity_id": entity_id, "features": features},
            timeout=30  # 30 second timeout
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            error = response.json()["error"]
            print(f"Bad Request: {error['message']}")
            # Handle validation errors (e.g., missing features)
            return None
        elif response.status_code == 401:
            print("Authentication failed: Check API key")
            # Refresh/rotate API key
            return None
        elif response.status_code == 404:
            print(f"Resource not found: {response.json()['error']['message']}")
            return None
        elif response.status_code == 429:
            # Rate limit exceeded
            error = response.json()["error"]
            retry_after = error["details"]["retry_after"]
            print(f"Rate limit exceeded. Retry after {retry_after} seconds")
            time.sleep(retry_after)
            return make_prediction(entity_id, features)  # Retry
        elif response.status_code >= 500:
            print(f"Server error: {response.status_code}. Retrying...")
            time.sleep(5)
            return make_prediction(entity_id, features)  # Retry
        else:
            print(f"Unexpected error: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print("Request timeout. Try again later.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error. Check network connectivity.")
        return None
    except Exception as e:
        print(f"Unexpected exception: {str(e)}")
        return None

# Usage
result = make_prediction("user_12345", {"tenure_days": 365, "engagement_score": 0.78})
if result:
    print(f"Prediction: {result['prediction']['score']}")
```

### Rate Limit Handling

**Rate Limits:**
- Standard endpoints (`/predict`, `/models`, `/explain`): **100 requests/minute**
- Batch endpoint (`/predict/batch`): **10 requests/minute**
- Health endpoint (`/health`): **Unlimited** (no authentication required)

**429 Error Response:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded: 100 requests per minute",
    "details": {
      "limit": 100,
      "window": "1 minute",
      "retry_after": 45
    }
  },
  "timestamp": "2026-01-21T20:07:00Z",
  "request_id": "req_xyz789"
}
```

**Handling Strategy:**
- Respect `retry_after` value in error response
- Implement exponential backoff for repeated 429s
- Use batch endpoint for bulk predictions (higher throughput)
- Cache predictions when possible to reduce API calls

---

**Document Control:**
- **Review Cycle:** Quarterly or on API changes
- **Approval Required:** API Owner, Tech Lead
- **Related Documents:** `technical_design.md`, `user_guide.md`
