# Project 2: Dockerized Model Serving with Kubernetes Orchestration

## Implementation Guide

**Author:** MLOps Portfolio Team  
**Last Updated:** January 2026  
**Target Audience:** MLOps Engineers, DevOps Engineers, Platform Engineers

---

## Executive Summary

This project demonstrates enterprise-grade machine learning model deployment using Docker containerization and Kubernetes orchestration. The implementation showcases production-ready practices for serving ML models at scale, including containerization best practices, health monitoring, auto-scaling, and Prometheus metrics integration. This project directly addresses the infrastructure and deployment requirements to operate large-scale ML systems.

The architecture is designed to be cloud-agnostic, supporting deployment on any Kubernetes-compatible platform including AWS EKS, Google GKE, Azure AKS, or on-premises Kubernetes clusters.

---

## Architecture Overview

### System Components

The ML model serving system consists of four primary layers:

**Application Layer:** A FastAPI-based REST API that wraps the trained ML model and exposes prediction endpoints. FastAPI provides automatic API documentation, request validation, and asynchronous request handling for high throughput.

**Containerization Layer:** Docker packages the application with all dependencies into a lightweight, reproducible container image. Multi-stage builds optimize image size and security by separating build dependencies from runtime dependencies.

**Orchestration Layer:** Kubernetes manages container deployment, scaling, and networking. Services expose the application to internal and external traffic, while Deployments ensure the desired number of replicas are always running.

**Monitoring Layer:** Prometheus metrics are exposed at `/metrics` endpoint, enabling real-time monitoring of application health, request rates, and model inference latency.

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Kubernetes Service (LoadBalancer)        │   │
│  │  - Exposes port 80/443 to external traffic      │   │
│  │  - Routes requests to Pod replicas              │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│     ┌────────────────────┼────────────────────┐         │
│     │                    │                    │         │
│  ┌──▼──┐            ┌──▼──┐            ┌──▼──┐        │
│  │ Pod │            │ Pod │            │ Pod │        │
│  │ Rep1│            │ Rep2│            │ Rep3│        │
│  │     │            │     │            │     │        │
│  │ API │            │ API │            │ API │        │
│  │     │            │     │            │     │        │
│  │Model│            │Model│            │Model│        │
│  └──┬──┘            └──┬──┘            └──┬──┘        │
│     │                  │                  │            │
│     └──────────────────┼──────────────────┘            │
│                        │                               │
│                  ┌─────▼─────┐                         │
│                  │ Prometheus │                         │
│                  │  Scraper   │                         │
│                  └────────────┘                         │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Technical Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **API Framework** | FastAPI | 0.100+ | High-performance REST API server |
| **ASGI Server** | Uvicorn | 0.23+ | Async application server |
| **Containerization** | Docker | 20.10+ | Container image building and execution |
| **Orchestration** | Kubernetes | 1.24+ | Container orchestration and management |
| **Monitoring** | Prometheus | 2.40+ | Metrics collection and storage |
| **Model Format** | MLflow Models | 2.x | Standardized model packaging |
| **HTTP Client** | Requests | 2.31+ | Internal service communication |
| **Logging** | Python logging | 3.11+ | Structured application logging |

---

## Project Structure

```
project2-k8s/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── models.py               # Pydantic request/response schemas
│   ├── inference.py            # Model loading and prediction logic
│   ├── health.py               # Health check endpoints
│   ├── metrics.py              # Prometheus metrics definitions
│   └── requirements.txt         # Python dependencies
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment manifest
│   ├── service.yaml            # Kubernetes Service manifest
│   ├── configmap.yaml          # Configuration management
│   ├── hpa.yaml                # Horizontal Pod Autoscaler
│   └── namespace.yaml          # Kubernetes namespace
├── Dockerfile                   # Multi-stage Docker build
├── .dockerignore                # Docker build context exclusions
├── docker-compose.yml           # Local development orchestration
├── scripts/
│   ├── build.sh                # Docker image build script
│   ├── push.sh                 # Docker image push script
│   └── deploy.sh               # Kubernetes deployment script
├── tests/
│   ├── test_api.py             # API endpoint tests
│   └── test_inference.py       # Model inference tests
└── README.md                    # Project documentation
```

---

## Implementation Details

### 1. FastAPI Application

**Application Entry Point (`app/main.py`):**

The FastAPI application serves as the REST API wrapper for the ML model:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
import time

app = FastAPI(
    title="ML Model Serving API",
    description="Production-grade ML model serving endpoint",
    version="1.0.0"
)

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_name', 'status']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Generate predictions for input features.
    
    Args:
        request: PredictionRequest containing model inputs
        
    Returns:
        PredictionResponse with predictions and confidence scores
    """
    start_time = time.time()
    
    try:
        # Load model (cached in memory)
        model = get_model()
        
        # Prepare features
        features = prepare_features(request.features)
        
        # Generate prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()
        
        # Record metrics
        prediction_latency.labels(model_name='default').observe(
            time.time() - start_time
        )
        prediction_counter.labels(
            model_name='default',
            status='success'
        ).inc()
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            latency_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        prediction_counter.labels(
            model_name='default',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Request/Response Schemas (`app/models.py`):**

Pydantic models ensure type safety and automatic request validation:

```python
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    features: List[float] = Field(
        ...,
        description="Input features for model",
        example=[1.0, 2.0, 3.0]
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0]
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    prediction: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
```

**Health Check Endpoints (`app/health.py`):**

Kubernetes uses health checks to determine pod readiness and liveness:

```python
from fastapi import APIRouter
from app.inference import is_model_loaded

router = APIRouter()

@router.get("/health/live")
async def liveness():
    """
    Liveness probe: indicates if the pod should be restarted.
    Returns 200 if the application is running.
    """
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness():
    """
    Readiness probe: indicates if the pod is ready to receive traffic.
    Returns 200 only if the model is loaded and ready.
    """
    if is_model_loaded():
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "reason": "model_not_loaded"}

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest
    return generate_latest()
```

### 2. Model Loading and Inference

**Inference Module (`app/inference.py`):**

Efficient model loading with caching to minimize latency:

```python
import mlflow.pyfunc
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the ML model.
    
    The @lru_cache decorator ensures the model is loaded only once
    and reused for all subsequent requests.
    """
    try:
        # Load from MLflow Model Registry
        model_uri = "models:/RandomForest-Classifier/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded successfully from {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def is_model_loaded():
    """Check if model is cached and available"""
    try:
        get_model()
        return True
    except:
        return False

def prepare_features(raw_features):
    """
    Prepare raw input features for model inference.
    
    This function should mirror the preprocessing applied during training
    to ensure consistent feature representation.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Convert to numpy array
    features = np.array(raw_features).reshape(1, -1)
    
    # Apply scaling (must use the same scaler as training)
    scaler = load_training_scaler()
    features_scaled = scaler.transform(features)
    
    return features_scaled
```

### 3. Docker Containerization

**Multi-Stage Dockerfile:**

The Dockerfile uses multi-stage builds to minimize image size and improve security:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY app/ .

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Change to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key Dockerfile Practices:**

- **Multi-stage builds:** Separates build dependencies from runtime, reducing final image size from ~500MB to ~200MB.
- **Non-root user:** Runs the application as a non-privileged user (`appuser`) to improve security.
- **Health checks:** Docker health checks enable Kubernetes to detect unhealthy containers.
- **Environment variables:** Configures Python to run in unbuffered mode for real-time logging.

**Building and Pushing the Image:**

```bash
# Build image
docker build -t ml-serving:1.0.0 .

# Tag for registry
docker tag ml-serving:1.0.0 registry.example.com/ml-serving:1.0.0

# Push to registry
docker push registry.example.com/ml-serving:1.0.0

# Verify image
docker images | grep ml-serving
```

### 4. Kubernetes Deployment

**Deployment Manifest (`k8s/deployment.yaml`):**

The Deployment ensures the desired number of replicas are always running:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-serving
  namespace: ml-platform
  labels:
    app: ml-serving
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ml-serving
  template:
    metadata:
      labels:
        app: ml-serving
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: registry.example.com/ml-serving:1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        
        # Resource requests and limits
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        
        # Liveness probe: restart if unhealthy
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Readiness probe: remove from load balancer if not ready
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        # Environment variables
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: ml-serving-config
              key: mlflow_uri
        - name: LOG_LEVEL
          value: "INFO"
```

**Service Manifest (`k8s/service.yaml`):**

The Service exposes the Deployment to internal and external traffic:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-serving
  namespace: ml-platform
  labels:
    app: ml-serving
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: ml-serving
  sessionAffinity: None
```

**Horizontal Pod Autoscaler (`k8s/hpa.yaml`):**

Automatically scales the number of replicas based on CPU utilization:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-serving-hpa
  namespace: ml-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

### 5. Deployment and Scaling

**Deploying to Kubernetes:**

```bash
# Create namespace
kubectl create namespace ml-platform

# Create ConfigMap
kubectl create configmap ml-serving-config \
  --from-literal=mlflow_uri=http://mlflow-tracking:5000 \
  -n ml-platform

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get deployments -n ml-platform
kubectl get pods -n ml-platform
kubectl get svc -n ml-platform
```

**Monitoring Pod Status:**

```bash
# Watch pod creation
kubectl get pods -n ml-platform -w

# Check pod logs
kubectl logs -n ml-platform deployment/ml-serving -f

# Describe pod for events
kubectl describe pod -n ml-platform <pod-name>

# Check resource usage
kubectl top pods -n ml-platform
```

### 6. Monitoring with Prometheus

**Prometheus Metrics:**

The application exposes metrics at `/metrics` endpoint in Prometheus format:

```
# HELP predictions_total Total number of predictions
# TYPE predictions_total counter
predictions_total{model_name="default",status="success"} 1523.0
predictions_total{model_name="default",status="error"} 12.0

# HELP prediction_latency_seconds Prediction latency in seconds
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{model_name="default",le="0.1"} 1450.0
prediction_latency_seconds_bucket{model_name="default",le="0.5"} 1510.0
prediction_latency_seconds_bucket{model_name="default",le="1.0"} 1520.0
prediction_latency_seconds_bucket{model_name="default",le="+Inf"} 1523.0
```

**Prometheus ServiceMonitor:**

Configure Prometheus to scrape metrics from the application:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-serving
  namespace: ml-platform
spec:
  selector:
    matchLabels:
      app: ml-serving
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

---

## Execution Guide

### Local Development

**Using Docker Compose:**

```bash
# Start services locally
docker-compose up -d

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'

# View API documentation
open http://localhost:8000/docs

# View logs
docker-compose logs -f api
```

### Production Deployment

**Step 1: Build and Push Image**

```bash
./scripts/build.sh
./scripts/push.sh
```

**Step 2: Deploy to Kubernetes**

```bash
./scripts/deploy.sh
```

**Step 3: Verify Deployment**

```bash
kubectl get all -n ml-platform
kubectl logs -n ml-platform deployment/ml-serving
```

**Step 4: Test Endpoint**

```bash
# Get service external IP
EXTERNAL_IP=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test prediction
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

---

## Performance Optimization

### Latency Optimization

**Model Caching:** The model is loaded once and cached in memory using `@lru_cache`, eliminating reload overhead for each request.

**Async Processing:** FastAPI with Uvicorn handles multiple concurrent requests efficiently using async/await patterns.

**Request Batching:** For high-throughput scenarios, implement batch prediction endpoints:

```python
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction for multiple samples"""
    all_features = [prepare_features(r.features) for r in requests]
    predictions = model.predict(np.vstack(all_features))
    return [{"prediction": p} for p in predictions]
```

### Resource Optimization

**CPU and Memory Limits:** Set appropriate resource requests and limits in the Deployment to ensure fair resource allocation across pods.

**Horizontal Scaling:** The HorizontalPodAutoscaler automatically scales replicas based on CPU and memory utilization, ensuring consistent performance under load.

---

## Troubleshooting

### Common Issues

**Issue: Pod CrashLoopBackOff**

*Diagnosis:* Check pod logs for errors.

```bash
kubectl logs -n ml-platform <pod-name>
```

*Solution:* Verify the model is available at the specified MLflow URI and all dependencies are correctly installed.

**Issue: High Latency**

*Diagnosis:* Check prediction latency metrics.

```bash
kubectl exec -n ml-platform <pod-name> -- curl http://localhost:8000/metrics | grep prediction_latency
```

*Solution:* Increase resource limits or scale up the number of replicas.

**Issue: Service Unreachable**

*Diagnosis:* Verify service and pod status.

```bash
kubectl get svc -n ml-platform
kubectl get pods -n ml-platform
```

*Solution:* Ensure the service type is LoadBalancer and the external IP is assigned.

---

## References

1. [FastAPI Documentation](https://fastapi.tiangolo.com/)
2. [Kubernetes Documentation - Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
3. [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
4. [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
5. [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

---

## Next Steps

1. **Add Ingress Controller:** Configure Kubernetes Ingress for HTTP/HTTPS routing and SSL termination.
2. **Implement Service Mesh:** Use Istio or Linkerd for advanced traffic management and observability.
3. **Add Model Versioning:** Implement canary deployments to gradually roll out new model versions.
4. **Integrate with CI/CD:** Automate deployment as part of a CI/CD pipeline (see Project 3).
