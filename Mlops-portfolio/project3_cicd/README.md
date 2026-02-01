# cicd: Complete CI/CD Pipeline for ML Lifecycle

This project demonstrates the automation of machine learning workflows using GitHub Actions. It implements Continuous Integration (CI) and Continuous Deployment (CD) principles specifically for ML models.

## Key Features
- **Continuous Integration (CI)**:
    - Automated linting with `flake8`.
    - Unit testing with `pytest` for model logic and data integrity.
- **Continuous Deployment (CD)**:
    - Automated Docker image builds on every push to the main branch.
- **Data Versioning**: Integrated concepts of DVC for tracking datasets (simulated).
- **Environment Management**: Automated dependency installation and environment setup in CI runners.

## Setup & Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests locally: `pytest tests/`
3. Trigger the pipeline: Push code to the `main` branch of your GitHub repository.

## Technical Stack
- GitHub Actions
- Pytest
- Docker
- DVC
- Flake8
- Scikit-learn
