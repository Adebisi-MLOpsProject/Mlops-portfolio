# MLOps Portfolio: Pipeline Projects 

This repository contains three hands-on MLOps projects. 

## Projects Overview

| Project | Focus | Key Technologies | Job Market Relevance |
| :--- | :--- | :--- | :--- |
| **Project 1** | Experiment Tracking | MLflow, Optuna, Scikit-learn | Model governance and reproducibility|
| **Project 2** | K8s Deployment | FastAPI, Docker, Kubernetes | Scalable infrastructure and serving|
| **Project 3** | CI/CD Pipeline | GitHub Actions, Pytest, DVC | Automation and GitOps |

## Project Details

### 1. End-to-End MLflow Experiment Tracking
Demonstrates how to manage the ML lifecycle by tracking experiments, tuning hyperparameters with Optuna, and versioning models in the MLflow Registry.
- **Key Outcome**: A reproducible training pipeline with documented metrics and artifacts.

### 2. Dockerized Model Serving with Kubernetes
Focuses on production-grade deployment. Includes a FastAPI wrapper for an ML model, multi-stage Docker builds, and Kubernetes manifests for orchestration and monitoring.
- **Key Outcome**: A scalable, monitored model serving endpoint ready for production traffic.

### 3. Complete CI/CD Pipeline for ML
Implements a full automation pipeline using GitHub Actions. Covers code quality (linting), unit testing for models, and automated container image builds.
- **Key Outcome**: A "push-to-deploy" workflow that ensures model and code integrity.

## How to Use
Each project is contained in its own directory with a dedicated README and setup instructions.
- `/project1_mlflow`: MLflow tracking and optimization.
- `/project2_k8s`: Docker and Kubernetes deployment.
- `/project3_cicd`: GitHub Actions and CI/CD automation.
