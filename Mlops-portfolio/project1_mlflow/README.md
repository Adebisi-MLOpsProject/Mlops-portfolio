# Project 1: End-to-End MLflow Experiment Tracking

This project demonstrates professional MLOps practices for experiment tracking and model management using MLflow. 

## Key Features
- **Experiment Tracking**: Automated logging of hyperparameters, metrics (RMSE, MAE, R2), and model artifacts.
- **Hyperparameter Optimization**: Integrated with Optuna for automated tuning of RandomForest models.
- **Model Registry**: Foundation for model versioning and lifecycle management.
- **Reproducibility**: Defined environment via `requirements.txt` and MLflow's standard model format.

## Setup & Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train.py`
3. View results: `mlflow ui`

## Technical Stack
- Python
- MLflow
- Scikit-learn
- Optuna
- Pandas
