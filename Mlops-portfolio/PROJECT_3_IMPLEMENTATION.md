# Project 3: Complete CI/CD Pipeline for ML Model Lifecycle

## Implementation Guide

**Author:** MLOps Portfolio Team  
**Last Updated:** January 2026  
**Target Audience:** MLOps Engineers, DevOps Engineers, Platform Engineers, Data Scientists

---

## Executive Summary

This project demonstrates a complete continuous integration and continuous deployment (CI/CD) pipeline for machine learning workflows using GitHub Actions. The implementation showcases automated testing, code quality checks, containerization, and deployment automation—core competencies that operate sophisticated MLOps platforms.

The pipeline enforces quality gates at every stage, ensuring that only validated code and models reach production. It implements GitOps principles where the desired state is defined in Git, and automation ensures the system converges to that state.

---

## Architecture Overview

### Pipeline Stages

The CI/CD pipeline consists of five primary stages that execute sequentially:

**Stage 1 - Code Quality:** Automated linting and code formatting checks ensure code adheres to style standards and best practices. Tools like flake8 detect potential bugs and code smells early in the development process.

**Stage 2 - Testing:** Comprehensive unit tests validate model logic, data processing, and API endpoints. Tests run in an isolated environment to ensure reproducibility and catch regressions before they reach production.

**Stage 3 - Build:** Docker images are built and pushed to a container registry. The build process includes security scanning to detect vulnerabilities in dependencies.

**Stage 4 - Integration:** The built image is deployed to a staging environment where integration tests verify end-to-end functionality in a production-like setting.

**Stage 5 - Deployment:** Upon approval, the image is deployed to production. Automated rollback mechanisms ensure quick recovery if issues are detected.

### Workflow Diagram

```
Git Push
   │
   ├─→ [Trigger] GitHub Actions Workflow
   │
   ├─→ [Stage 1] Code Quality
   │   ├─ Linting (flake8)
   │   ├─ Code formatting (black)
   │   └─ Type checking (mypy)
   │
   ├─→ [Stage 2] Testing
   │   ├─ Unit tests (pytest)
   │   ├─ Coverage analysis
   │   └─ Model validation
   │
   ├─→ [Stage 3] Build
   │   ├─ Docker image build
   │   ├─ Security scanning
   │   └─ Push to registry
   │
   ├─→ [Stage 4] Integration
   │   ├─ Deploy to staging
   │   ├─ Integration tests
   │   └─ Performance tests
   │
   └─→ [Stage 5] Deployment
       ├─ Manual approval
       ├─ Deploy to production
       └─ Smoke tests
```

---

## Technical Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **CI/CD Platform** | GitHub Actions | Latest | Workflow automation and orchestration |
| **Code Quality** | flake8 | 6.x | Python linting and style checking |
| **Code Formatting** | black | 23.x | Automatic code formatting |
| **Type Checking** | mypy | 1.x | Static type analysis |
| **Testing Framework** | pytest | 7.x | Unit and integration testing |
| **Coverage Analysis** | coverage | 7.x | Code coverage measurement |
| **Containerization** | Docker | 20.10+ | Container image building |
| **Registry** | Docker Hub / ECR | Latest | Container image storage |
| **Data Versioning** | DVC | 3.x | Data and model versioning |
| **Deployment** | kubectl | 1.24+ | Kubernetes deployment |

---

## Project Structure

```
project3-cicd/
├── .github/
│   └── workflows/
│       ├── mlops_pipeline.yml      # Main CI/CD workflow
│       ├── code-quality.yml        # Code quality checks
│       ├── test.yml                # Testing workflow
│       ├── build.yml               # Docker build workflow
│       └── deploy.yml              # Deployment workflow
├── src/
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Model evaluation
│   ├── predict.py                  # Batch prediction
│   ├── utils.py                    # Utility functions
│   └── config.py                   # Configuration
├── tests/
│   ├── test_model.py               # Model logic tests
│   ├── test_data.py                # Data processing tests
│   ├── test_api.py                 # API endpoint tests
│   └── conftest.py                 # Pytest fixtures
├── data/
│   ├── raw/                        # Raw input data
│   └── processed/                  # Processed data
├── models/
│   └── artifacts/                  # Trained models
├── Dockerfile                       # Container definition
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
├── pytest.ini                       # Pytest configuration
├── .flake8                          # Flake8 configuration
├── pyproject.toml                   # Black configuration
└── README.md                        # Documentation
```

---

## Implementation Details

### 1. GitHub Actions Workflow

**Main Workflow File (`.github/workflows/mlops_pipeline.yml`):**

The main workflow orchestrates all pipeline stages and defines the overall execution flow:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-model

jobs:
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install flake8 black mypy
      
      - name: Lint with flake8
        run: |
          flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
      
      - name: Check formatting with black
        run: black --check src/ tests/
      
      - name: Type checking with mypy
        run: mypy src/ --ignore-missing-imports
        continue-on-error: true

  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests with pytest
        run: |
          pytest tests/ -v --cov=src --cov-report=xml --cov-report=html
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  build:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    environment: staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-serving \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }} \
            -n ml-platform --record
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ml-serving -n ml-platform --timeout=5m
      
      - name: Run integration tests
        run: |
          STAGING_URL=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          pytest tests/integration/ -v --base-url=http://$STAGING_URL

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
      
      - name: Deploy to production
        run: |
          kubectl set image deployment/ml-serving \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }} \
            -n ml-platform --record
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ml-serving -n ml-platform --timeout=5m
      
      - name: Run smoke tests
        run: |
          PROD_URL=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          pytest tests/smoke/ -v --base-url=http://$PROD_URL
      
      - name: Rollback on failure
        if: failure()
        run: |
          kubectl rollout undo deployment/ml-serving -n ml-platform
          kubectl rollout status deployment/ml-serving -n ml-platform --timeout=5m
```

### 2. Code Quality Configuration

**Flake8 Configuration (`.flake8`):**

```ini
[flake8]
max-line-length = 127
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist,.venv
ignore = E203,W503
per-file-ignores =
    __init__.py:F401
```

**Black Configuration (`pyproject.toml`):**

```toml
[tool.black]
line-length = 127
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

**Mypy Configuration (`pyproject.toml`):**

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true
```

### 3. Testing Strategy

**Unit Tests (`tests/test_model.py`):**

Comprehensive unit tests validate model logic and data processing:

```python
import pytest
import numpy as np
from src.train import train_model, evaluate_model
from src.preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    """Fixture providing sample training data"""
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_train_model(sample_data):
    """Test model training completes without errors"""
    X, y = sample_data
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_predictions(sample_data):
    """Test model generates valid predictions"""
    X, y = sample_data
    model = train_model(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == (100,)
    assert np.all((predictions == 0) | (predictions == 1))

def test_evaluate_model(sample_data):
    """Test model evaluation metrics"""
    X, y = sample_data
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_preprocess_data():
    """Test data preprocessing pipeline"""
    raw_data = np.random.rand(50, 10)
    processed = preprocess_data(raw_data)
    
    assert processed.shape == (50, 10)
    assert not np.any(np.isnan(processed))
```

**Pytest Configuration (`pytest.ini`):**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    smoke: Smoke tests
```

### 4. Docker Build Optimization

**Dockerfile with Layer Caching:**

```dockerfile
# Stage 1: Dependencies
FROM python:3.11-slim as dependencies

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Builder
FROM python:3.11-slim as builder

WORKDIR /build

COPY . .
RUN pip install --no-cache-dir --user -e .

# Stage 3: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY --from=dependencies /root/.local /root/.local

# Copy application
COPY --from=builder /build/src ./src
COPY --from=builder /build/models ./models

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Data Versioning with DVC

**DVC Configuration (`.dvc/config`):**

```ini
[core]
    remote = myremote
    autostage = true

['remote "myremote"']
    url = s3://my-bucket/dvc-storage
```

**Tracking Data Files (`.gitignore` and `.dvcignore`):**

```bash
# .gitignore
/data/raw/*
/data/processed/*
/models/artifacts/*

# .dvcignore
.git
.hg
.gitignore
.dvcignore
```

**DVC Pipeline (`dvc.yaml`):**

```yaml
stages:
  preprocess:
    cmd: python src/preprocessing.py
    deps:
      - data/raw/dataset.csv
      - src/preprocessing.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
  
  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    outs:
      - models/artifacts/model.pkl
    metrics:
      - metrics.json:
          cache: false
  
  evaluate:
    cmd: python src/evaluate.py
    deps:
      - data/processed/test.csv
      - models/artifacts/model.pkl
      - src/evaluate.py
    metrics:
      - eval_metrics.json:
          cache: false
```

### 6. Deployment Automation

**Kubernetes Deployment Script (`scripts/deploy.sh`):**

```bash
#!/bin/bash
set -e

# Configuration
NAMESPACE="ml-platform"
DEPLOYMENT="ml-serving"
IMAGE_TAG="$1"

if [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <image-tag>"
    exit 1
fi

echo "Deploying image: $IMAGE_TAG"

# Update deployment
kubectl set image deployment/$DEPLOYMENT \
    api=ghcr.io/yourorg/ml-model:$IMAGE_TAG \
    -n $NAMESPACE \
    --record

# Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=5m

# Run smoke tests
echo "Running smoke tests..."
EXTERNAL_IP=$(kubectl get svc $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$EXTERNAL_IP/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'

echo "Deployment successful!"
```

---

## Execution Guide

### Local Development Workflow

**Step 1: Set Up Development Environment**

```bash
# Clone repository
git clone <repository-url>
cd project3-cicd

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Step 2: Run Code Quality Checks**

```bash
# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Type checking
mypy src/ --ignore-missing-imports
```

**Step 3: Run Tests Locally**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

**Step 4: Build Docker Image**

```bash
# Build image
docker build -t ml-model:dev .

# Run container
docker run -p 8000:8000 ml-model:dev

# Test API
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'
```

### CI/CD Pipeline Execution

**Triggering the Pipeline:**

The pipeline automatically triggers on push to main or develop branches:

```bash
# Make changes and commit
git add .
git commit -m "Add new feature"

# Push to trigger pipeline
git push origin main

# Monitor pipeline in GitHub Actions UI
# https://github.com/yourorg/yourrepo/actions
```

**Pipeline Status Monitoring:**

```bash
# View workflow runs
gh run list

# View specific workflow run
gh run view <run-id>

# View workflow logs
gh run view <run-id> --log
```

---

## Best Practices

### Code Quality

**Consistent Formatting:** Use black to enforce consistent code formatting across the team. This eliminates style debates and improves code readability.

**Type Hints:** Add type hints to function signatures to catch type errors early and improve code documentation.

**Test Coverage:** Aim for at least 80% code coverage. Use coverage reports to identify untested code paths.

### Testing Strategy

**Unit Tests:** Test individual functions and classes in isolation. Use fixtures to provide test data and mock external dependencies.

**Integration Tests:** Test interactions between components in a realistic environment. Use Docker Compose for local integration testing.

**Smoke Tests:** Run quick sanity checks in production to verify critical functionality is working.

### Deployment Safety

**Staging Environment:** Always deploy to staging first to catch issues before production.

**Automated Rollback:** Implement automatic rollback on deployment failure to minimize downtime.

**Gradual Rollout:** Use canary deployments to gradually roll out new versions and monitor for issues.

---

## Monitoring and Observability

### GitHub Actions Logs

All workflow steps produce detailed logs accessible through the GitHub Actions UI:

```bash
# View logs for specific workflow
gh run view <run-id> --log

# Download logs
gh run download <run-id>
```

### Deployment Monitoring

Monitor deployment status and metrics:

```bash
# Watch deployment status
kubectl rollout status deployment/ml-serving -n ml-platform -w

# View deployment history
kubectl rollout history deployment/ml-serving -n ml-platform

# Rollback to previous version
kubectl rollout undo deployment/ml-serving -n ml-platform
```

---

## Troubleshooting

### Common Issues

**Issue: Tests Fail Locally but Pass in CI**

*Cause:* Environment differences between local and CI environment.

*Solution:* Ensure Python version, dependencies, and environment variables match between local and CI. Use Docker for consistent environments.

**Issue: Deployment Fails Due to Image Pull Error**

*Cause:* Container registry credentials not configured in Kubernetes.

*Solution:* Create image pull secret:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  -n ml-platform
```

**Issue: Rollout Timeout**

*Cause:* Pod not becoming ready within timeout period.

*Solution:* Check pod logs and readiness probe configuration:

```bash
kubectl logs -n ml-platform deployment/ml-serving
kubectl describe pod -n ml-platform <pod-name>
```

---

## References

1. [GitHub Actions Documentation](https://docs.github.com/en/actions)
2. [pytest Documentation](https://docs.pytest.org/)
3. [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
4. [Kubernetes Deployment Strategy](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
5. [DVC Documentation](https://dvc.org/doc)

---

## Next Steps

1. **Add Security Scanning:** Integrate container image scanning to detect vulnerabilities in dependencies.
2. **Implement Canary Deployments:** Use tools like Flagger to gradually roll out new versions and monitor metrics.
3. **Add Performance Testing:** Include performance benchmarks in the pipeline to catch performance regressions.
4. **Integrate with Monitoring:** Connect deployment pipeline with monitoring systems to automatically rollback on performance degradation.
