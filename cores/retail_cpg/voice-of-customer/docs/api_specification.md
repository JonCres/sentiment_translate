# API Specification

**AI Core:** `Voice of Customer AI`  
**Industry:** `Retail & CPG`  
**API Version:** `v1.0.0`  
**OpenAPI Version:** `3.0.3`  
**Last Updated:** `2026-01-22`

---

## OpenAPI Specification

```yaml
openapi: 3.0.3
info:
  title: "Voice of Customer AI API"
  description: |
    RESTful API for the Voice of Customer AI AI Core.
    
    **Capabilities:**
    - Single interaction analysis (Sentiment, Topics, ABSA, Emotions)
    - Batch analysis
    - Model metadata retrieval
    - Feature importance explanation
    
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
  - name: "Analysis"
    description: "Core VoC analysis endpoints (Sentiment, Topics, ABSA)"
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

  /analyze:
    post:
      tags:
        - "Analysis"
      summary: "Single interaction analysis"
      description: |
        Process a text interaction (review, chat log, survey) and return unified insights:
        Sentiment, Topics, Aspects, Emotions, and Recommended Actions.
      operationId: "analyzeInteraction"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AnalysisRequest"
            example:
              interaction_id: "review_12345"
              payload: "The product quality is amazing, but the shipping was delayed by 3 days. I am frustrated."
              customer_id: "cust_5678"
              metadata:
                channel: "web_review"
                product_id: "sku_999"
      responses:
        "200":
          description: "Analysis successful"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/UnifiedInteractionResponse"
              example:
                interaction_id: "review_12345"
                sentiment:
                  label: "neutral"
                  confidence: 0.85
                  score: 0.12
                topics:
                  - id: 10
                    name: "Shipping Delays"
                    probability: 0.72
                  - id: 5
                    name: "Product Quality"
                    probability: 0.65
                aspects:
                  - aspect: "quality"
                    sentiment: "positive"
                    confidence: 0.98
                  - aspect: "shipping"
                    sentiment: "negative"
                    confidence: 0.95
                emotions:
                  - emotion: "frustration"
                    score: 0.88
                  - emotion: "joy"
                    score: 0.45
                insights:
                  urgency: "Medium"
                  recommended_action: "Check Logistics / Delivery Status"
        "400":
          $ref: "#/components/responses/BadRequest"
        "401":
          $ref: "#/components/responses/Unauthorized"
        "429":
          $ref: "#/components/responses/RateLimitExceeded"
        "500":
          $ref: "#/components/responses/InternalServerError"

  /analyze/batch:
    post:
      tags:
        - "Analysis"
      summary: "Batch analysis"
      description: |
        Process multiple interactions in batch.
        
        **Limits:**
        - Maximum 1000 entities per batch
        - Asynchronous processing for batches > 100 entities
      operationId: "analyzeBatch"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/BatchAnalysisRequest"
            example:
              interactions:
                - interaction_id: "review_1"
                  payload: "Great!"
                - interaction_id: "review_2"
                  payload: "Terrible."
              async: true
      responses:
        "202":
          description: "Batch job accepted (async)"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BatchJobResponse"
        "200":
          description: "Batch completed (sync)"
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BatchAnalysisResponse"
        "400":
          $ref: "#/components/responses/BadRequest"
        "401":
          $ref: "#/components/responses/Unauthorized"

  /models:
    get:
      tags:
        - "Models"
      summary: "List active models"
      description: "Retrieve metadata for all registered models (Sentiment, Topic, ABSA)"
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
                  - model_id: "absa_v1"
                    model_type: "absa"
                    status: "active"
                    version: "1.1.0"
                  - model_id: "topic_v2"
                    model_type: "bertopic"
                    status: "active"
                    version: "2.3.1"

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

    AnalysisRequest:
      type: "object"
      required:
        - "interaction_id"
        - "payload"
      properties:
        interaction_id:
          type: "string"
          description: "Unique identifier for the interaction"
        payload:
          type: "string"
          description: "The text content to analyze (review, transcript, etc.)"
        customer_id:
          type: "string"
          description: "Optional customer identifier"
        metadata:
          type: "object"
          description: "Additional context (channel, product_id, etc.)"

    UnifiedInteractionResponse:
      type: "object"
      properties:
        interaction_id:
          type: "string"
        sentiment:
          type: "object"
          properties:
            label: 
              type: "string"
              enum: ["positive", "neutral", "negative"]
            confidence: 
              type: "number"
            score:
              type: "number"
        topics:
          type: "array"
          items:
            type: "object"
            properties:
              id:
                type: "integer"
              name:
                type: "string"
              probability:
                type: "number"
        aspects:
          type: "array"
          items:
            type: "object"
            properties:
              aspect:
                type: "string"
              sentiment:
                type: "string"
              confidence:
                type: "number"
        emotions:
          type: "array"
          items:
            type: "object"
            properties:
              emotion:
                type: "string"
              score:
                type: "number"
        insights:
          type: "object"
          properties:
            urgency:
              type: "string"
              enum: ["Low", "Medium", "High"]
            recommended_action:
              type: "string"

    BatchAnalysisRequest:
      type: "object"
      required:
        - "interactions"
      properties:
        interactions:
          type: "array"
          items:
            $ref: "#/components/schemas/AnalysisRequest"
          maxItems: 1000
        async:
          type: "boolean"
          default: true
          description: "Process asynchronously (recommended for >100 entities)"

    BatchAnalysisResponse:
      type: "object"
      properties:
        results:
          type: "array"
          items:
            $ref: "#/components/schemas/UnifiedInteractionResponse"
        metadata:
          type: "object"
          properties:
            processed_count:
              type: "integer"
            duration_ms:
              type: "number"

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
              version:
                type: "string"

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
              message: "Missing required field: interaction_id"
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
```bash
curl -H "X-API-Key: your_api_key_here" https://api.example.com/v1/health
```

### Single Analysis
```bash
curl -X POST https://api.example.com/v1/analyze \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "interaction_id": "review_12345",
    "payload": "The product quality is amazing, but the shipping was delayed by 3 days. I am frustrated.",
    "customer_id": "cust_5678"
  }'
```

### Batch Analysis
```bash
curl -X POST https://api.example.com/v1/analyze/batch \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "interactions": [
      {"interaction_id": "rev_1", "payload": "Great!"},
      {"interaction_id": "rev_2", "payload": "Terrible."}
    ],
    "async": true
  }'
```

---

**Document Control:**
- **Review Cycle:** Quarterly or on API changes
- **Approval Required:** API Owner, Tech Lead
- **Related Documents:** `technical_design.md`, `user_guide.md`

```