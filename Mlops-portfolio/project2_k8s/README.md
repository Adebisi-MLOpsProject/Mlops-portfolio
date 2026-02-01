# k8s project: Dockerized ML Model Deployment with Kubernetes

This project focuses on the "Ops" part of MLOps, demonstrating how to containerize an ML model and orchestrate its deployment using Kubernetes. This directly targets requirements for scalable ML infrastructure.

## Key Features
- **Containerization**: Optimized Dockerfile for a FastAPI-based model serving application.
- **Kubernetes Orchestration**: Complete manifests for Deployments, Services, and Resource management.
- **Scalability**: Configured for multiple replicas with health probes (Liveness/Readiness).
- **Monitoring**: Integrated Prometheus metrics endpoint for tracking prediction counts and latency.

## Setup & Usage
1. Build the Docker image: `docker build -t ml-model-api:latest .`
2. Apply Kubernetes manifests: `kubectl apply -f k8s/`
3. Access the API: `curl http://localhost/predict -d '{"features": [...]}'`
4. View metrics: `http://localhost/metrics`

## Technical Stack
- FastAPI
- Docker
- Kubernetes
- Prometheus
- Scikit-learn
