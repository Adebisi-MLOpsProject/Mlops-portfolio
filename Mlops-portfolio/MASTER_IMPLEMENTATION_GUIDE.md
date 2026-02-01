# MLOps Portfolio: Master Implementation Guide

## Complete ML Lifecycle Architecture

**Author:** Adebisi Ayokunle
**Last Updated:** January 2026 
**Audience:** MLOps Engineers, Platform Engineers, Hiring Managers

---

## Overview

This master guide provides a comprehensive view of how the three MLOps projects work together to form a complete, production-ready machine learning operations platform. Each project addresses a critical phase of the ML lifecycle: experimentation, deployment, and automation.

The architecture demonstrates enterprise-grade practices for managing the entire ML lifecycle from initial model development through production serving and continuous improvement.

---

## Complete ML Lifecycle Architecture

### The Three Pillars of MLOps

The portfolio demonstrates three fundamental pillars of modern MLOps:

**Pillar 1: Experiment Management (Project 1)**

Model development requires systematic experimentation with different algorithms, hyperparameters, and data preprocessing techniques. Project 1 showcases how to track experiments, manage model versions, and maintain reproducibility using MLflow.

**Pillar 2: Model Deployment (Project 2)**

Once a model is validated, it must be served reliably at scale. Project 2 demonstrates containerization with Docker and orchestration with Kubernetes, enabling high-availability, auto-scaling model serving infrastructure.

**Pillar 3: Automation & Governance (Project 3)**

Production ML systems require automated quality gates, testing, and deployment workflows. Project 3 implements a complete CI/CD pipeline that ensures only validated code and models reach production.

### End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ML LIFECYCLE ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: EXPERIMENTATION (Project - MLflow)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Raw Data → Preprocessing → Feature Engineering → Model Training            │
│     ↓           ↓                ↓                    ↓                       │
│  Logged      Logged          Logged              MLflow Tracking             │
│                                                      ↓                       │
│                                              Hyperparameter Optimization     │
│                                              (Optuna Integration)            │
│                                                      ↓                       │
│                                              Model Registry                  │
│                                              (Versioning & Staging)          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DEPLOYMENT (Project Kubernetes)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Model Registry → Docker Build → Container Registry → Kubernetes Deploy     │
│                                                              ↓               │
│                                                    Service Mesh              │
│                                                    Load Balancing            │
│                                                    Auto-scaling              │
│                                                    Health Monitoring         │
│                                                              ↓               │
│                                                    Production Serving        │
│                                                    (REST API Endpoints)      │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: AUTOMATION & GOVERNANCE (Project 3 - CI/CD)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Git Push → Code Quality → Testing → Build → Integration → Production       │
│     ↓          ↓            ↓        ↓         ↓              ↓             │
│  Trigger   Linting      Unit Tests  Docker   Staging      Deployment       │
│  Workflow  Type Check   Coverage    Image    Tests        Automation       │
│            Formatting   Validation  Push     Validation   Rollback         │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ↓
                            Production ML System
                            (Continuous Monitoring)
```

---

## Project Integration Points

### How Projects Connect

**Project 1 → Project 2 Connection:**

Project 1 produces a trained model registered in MLflow's Model Registry. Project 2 consumes this model by loading it from the registry and wrapping it in a FastAPI application that is containerized and deployed to Kubernetes.

```python
# In Project 2 (app/inference.py)
import mlflow.pyfunc

def get_model():
    # Load model from Project 1's MLflow Registry
    model_uri = "models:/RandomForest-Classifier/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
```

**Project 2 → Project 3 Connection:**

Project 2's Docker image is built and pushed to a container registry. Project 3's CI/CD pipeline automatically deploys this image to Kubernetes, replacing the previous version with zero downtime.

```yaml
# In Project 3 (.github/workflows/mlops_pipeline.yml)
- name: Deploy to production
  run: |
    kubectl set image deployment/ml-serving \
      api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }} \
      -n ml-platform --record
```

**Project 3 → Project 1 Connection:**

Project 3's CI/CD pipeline can trigger retraining workflows in Project 1 when new data arrives or model performance degrades. This creates a feedback loop for continuous model improvement.

```yaml
# In Project 3 (.github/workflows/retrain.yml)
- name: Trigger model retraining
  if: model_performance_degraded
  run: |
    python src/train.py --trigger-optuna --n-trials 50
    mlflow register_model --best-run
```

---

## Technology Stack Overview

| Layer | Project 1 | Project 2 | Project 3 | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Experimentation** | MLflow, Optuna | — | — | Track experiments and optimize hyperparameters |
| **Model Registry** | MLflow Registry | — | — | Version and stage models |
| **Application** | — | FastAPI | — | REST API for model serving |
| **Containerization** | — | Docker | Docker | Package applications for deployment |
| **Orchestration** | — | Kubernetes | Kubernetes | Manage containers at scale |
| **Automation** | — | — | GitHub Actions | Automate workflows and deployments |
| **Testing** | pytest | pytest | pytest | Validate code and models |
| **Monitoring** | — | Prometheus | — | Observe system health and performance |
| **Data Versioning** | — | — | DVC | Track data and model versions |

---

## Deployment Scenarios

### Scenario 1: New Model Development

**Timeline:** Data Scientist develops and validates a new model

1. **Data Scientist** runs `python src/train.py` in Project 1
2. **MLflow** tracks all experiments, parameters, and metrics
3. **Optuna** optimizes hyperparameters automatically
4. **Best model** is registered in MLflow's Model Registry
5. **Staging stage** is assigned for pre-production validation

### Scenario 2: Model Deployment to Production

**Timeline:** Model moves from staging to production

1. **MLOps Engineer** transitions model from Staging to Production in Project 1
2. **Project 2** loads the Production model from MLflow Registry
3. **Docker image** is built with the new model
4. **Project 3** CI/CD pipeline detects the new image
5. **Kubernetes** deployment is updated with zero downtime
6. **Monitoring** tracks model performance in production

### Scenario 3: Code Update and Redeployment

**Timeline:** Bug fix or feature addition to serving code

1. **Engineer** commits code change to Git
2. **Project 3** CI/CD pipeline triggers automatically
3. **Code quality checks** validate the changes
4. **Tests** ensure functionality is preserved
5. **Docker image** is built with updated code
6. **Staging deployment** validates the changes
7. **Production deployment** rolls out the update
8. **Automatic rollback** occurs if issues are detected

### Scenario 4: Model Performance Degradation

**Timeline:** Production model performance drops below threshold

1. **Monitoring system** detects performance degradation
2. **Alert** is sent to MLOps team
3. **Project 3** triggers retraining workflow
4. **Project 1** trains new model with recent data
5. **New model** is registered and staged
6. **Validation** confirms improvement over production model
7. **Deployment** proceeds if validation passes
8. **Rollback** occurs if new model doesn't improve performance

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Objective:** Set up core infrastructure and basic workflows

- [ ] Deploy MLflow tracking server (Project 1)
- [ ] Set up Docker registry (Project 2)
- [ ] Configure Kubernetes cluster (Project 2)
- [ ] Initialize GitHub Actions workflows (Project 3)

**Deliverables:**
- MLflow tracking server accessible at `mlflow.example.com`
- Kubernetes cluster with 3+ nodes
- Docker images building and pushing successfully
- GitHub Actions workflows executing on push

### Phase 2: Integration (Weeks 3-4)

**Objective:** Connect projects and implement end-to-end workflows

- [ ] Configure Project 1 to register models to MLflow Registry
- [ ] Update Project 2 to load models from MLflow Registry
- [ ] Implement Docker builds in Project 3 CI/CD
- [ ] Deploy Project 2 to Kubernetes via Project 3

**Deliverables:**
- Models automatically registered from Project 1
- Project 2 serving models from MLflow Registry
- Automated Docker builds and pushes
- Kubernetes deployments triggered by CI/CD

### Phase 3: Automation (Weeks 5-6)

**Objective:** Implement automated monitoring and retraining

- [ ] Set up Prometheus monitoring (Project 2)
- [ ] Implement performance monitoring alerts
- [ ] Configure automated retraining triggers (Project 3)
- [ ] Implement automatic rollback mechanisms

**Deliverables:**
- Prometheus metrics collected and visualized
- Alerts triggered on performance degradation
- Automated retraining on schedule or trigger
- Automatic rollback on deployment failure

### Phase 4: Production Hardening (Weeks 7-8)

**Objective:** Ensure production readiness and reliability

- [ ] Implement comprehensive logging and tracing
- [ ] Set up disaster recovery procedures
- [ ] Conduct load testing and optimization
- [ ] Document runbooks and troubleshooting guides

**Deliverables:**
- Centralized logging system (ELK stack or similar)
- Disaster recovery procedures documented
- Load testing results and optimization recommendations
- Comprehensive runbooks for common scenarios

---

## Key Metrics and Success Criteria

### Project 1: Experimentation Efficiency

| Metric | Target | Measurement |
| :--- | :--- | :--- |
| **Experiment Tracking** | 100% of runs logged | MLflow UI experiment count |
| **Model Reproducibility** | 100% of models reproducible | Ability to reload and rerun any model |
| **Hyperparameter Optimization** | 50% improvement over baseline | Best trial metric vs baseline |
| **Time to Best Model** | < 2 hours | Wall-clock time for optimization |

### Project 2: Deployment Reliability

| Metric | Target | Measurement |
| :--- | :--- | :--- |
| **Availability** | 99.9% uptime | Kubernetes pod uptime |
| **Latency** | < 100ms p99 | Prometheus prediction_latency_seconds |
| **Throughput** | > 1000 req/s | Requests per second under load |
| **Scaling Time** | < 2 minutes | Time from scale trigger to new pod ready |

### Project 3: Deployment Automation

| Metric | Target | Measurement |
| :--- | :--- | :--- |
| **Code Quality** | 100% pass rate | GitHub Actions code quality checks |
| **Test Coverage** | > 80% | Coverage.py report |
| **Deployment Frequency** | Daily | Deployments per day |
| **Mean Time to Recovery** | < 5 minutes | Time from failure to rollback complete |

---

## Security Considerations

### Authentication & Authorization

**MLflow Registry Access:**
- Implement RBAC to control who can promote models to Production
- Audit all model transitions between stages
- Require approval for Production stage transitions

**Kubernetes Access:**
- Use RBAC to limit deployment permissions
- Implement pod security policies
- Use network policies to restrict traffic

**CI/CD Pipeline:**
- Store secrets in GitHub Secrets, not in code
- Use short-lived credentials for deployments
- Implement approval gates for production deployments

### Data Security

**Model Artifacts:**
- Encrypt models at rest in the registry
- Use TLS for model transmission
- Implement access logging for model downloads

**Inference Data:**
- Use TLS for all API communications
- Implement rate limiting to prevent abuse
- Log all prediction requests for audit trails

### Container Security

**Image Scanning:**
- Scan all images for vulnerabilities before deployment
- Use minimal base images to reduce attack surface
- Keep dependencies up to date

**Runtime Security:**
- Run containers as non-root users
- Use read-only filesystems where possible
- Implement network policies to restrict pod communication

---

## Monitoring and Observability

### Metrics Collection

**Project 1 Metrics:**
- Experiment count and duration
- Model performance metrics (accuracy, precision, recall)
- Hyperparameter optimization progress
- Model registry transitions

**Project 2 Metrics:**
- Request rate and latency
- Error rate and error types
- Resource utilization (CPU, memory)
- Pod scaling events

**Project 3 Metrics:**
- Workflow execution time
- Test coverage and pass rate
- Deployment frequency and success rate
- Mean time to recovery

### Logging Strategy

**Structured Logging:**
- Use JSON format for all logs
- Include request IDs for tracing
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)

**Log Aggregation:**
- Centralize logs from all components
- Implement retention policies
- Enable full-text search for troubleshooting

**Alerting:**
- Alert on critical errors
- Alert on performance degradation
- Alert on deployment failures

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Model Not Found in MLflow Registry**

*Diagnosis:*
```bash
mlflow models list
mlflow models get-latest-versions --name RandomForest-Classifier
```

*Solution:*
1. Verify Project 1 training completed successfully
2. Check MLflow tracking server is accessible
3. Ensure model was registered with correct name

**Issue: Kubernetes Pod CrashLoopBackOff**

*Diagnosis:*
```bash
kubectl logs -n ml-platform deployment/ml-serving
kubectl describe pod -n ml-platform <pod-name>
```

*Solution:*
1. Check application logs for errors
2. Verify model is accessible from the pod
3. Ensure resource limits are not too restrictive

**Issue: CI/CD Pipeline Failure**

*Diagnosis:*
```bash
gh run view <run-id> --log
```

*Solution:*
1. Check specific workflow step that failed
2. Verify all dependencies are installed
3. Ensure secrets are configured correctly

---

## Best Practices Summary

### Development Practices

- **Version Everything:** Code, data, models, and configurations should all be versioned
- **Automate Testing:** Run tests automatically on every code change
- **Code Review:** Require peer review before merging to main branch
- **Documentation:** Keep documentation up to date with code changes

### Deployment Practices

- **Infrastructure as Code:** Define all infrastructure in version-controlled files
- **Immutable Deployments:** Use container images as immutable deployment units
- **Blue-Green Deployments:** Run old and new versions in parallel before switching
- **Automated Rollback:** Automatically rollback on deployment failure

### Operational Practices

- **Monitoring First:** Instrument systems with comprehensive monitoring before issues occur
- **Runbooks:** Document procedures for common operational tasks
- **Incident Response:** Define clear procedures for responding to incidents
- **Continuous Improvement:** Regularly review metrics and optimize processes

---

## References

1. [MLflow Documentation](https://mlflow.org/docs/latest/)
2. [Kubernetes Documentation](https://kubernetes.io/docs/)
3. [GitHub Actions Documentation](https://docs.github.com/en/actions)
4. [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
5. [Prometheus Monitoring](https://prometheus.io/docs/)
6. [MLOps.community Best Practices](https://mlops.community/)

---

## Next Steps

1. **Deploy the Architecture:** Follow the implementation roadmap to deploy all three projects
2. **Customize for Your Environment:** Adapt configurations for your specific infrastructure
3. **Train Your Team:** Ensure team members understand each component and how they integrate
4. **Iterate and Improve:** Continuously monitor metrics and optimize based on learnings
5. **Scale Gradually:** Start with a single model and gradually expand to multiple models

---

## Appendix: Quick Reference

### Project 1 Quick Start

```bash
cd project1-mlflow
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mlflow ui &
python src/train.py
```

### Project 2 Quick Start

```bash
cd project2-k8s
docker build -t ml-serving:1.0.0 .
docker-compose up -d
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### Project 3 Quick Start

```bash
cd project3-cicd
pip install -r requirements-dev.txt
pytest tests/ -v
flake8 src/ tests/
black src/ tests/
```

---

**For detailed implementation guides, refer to:**
- `PROJECT_1_IMPLEMENTATION.md` - MLflow Experiment Tracking
- `PROJECT_2_IMPLEMENTATION.md` - Kubernetes Deployment
- `PROJECT_3_IMPLEMENTATION.md` - CI/CD Pipeline
