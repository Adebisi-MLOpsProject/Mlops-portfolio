# Project 1: End-to-End MLflow Experiment Tracking

## Implementation Guide

**Author:** MLOps Portfolio Team  
**Last Updated:** January 2026  
**Target Audience:** MLOps Engineers, ML Platform Teams, DevOps Engineers

---

## Executive Summary

This project demonstrates professional machine learning operations practices using MLflow, a leading open-source platform for managing the complete ML lifecycle. The implementation showcases experiment tracking, hyperparameter optimization, model versioning, and reproducibility—core competencies that operate production ML platforms.

The project is designed to be production-ready, with clear separation of concerns, comprehensive logging, and best practices for model governance. It serves as a reference architecture for teams building scalable ML systems.

---

## Architecture Overview

### System Components

The MLflow experiment tracking system consists of three primary layers:

**Data Layer:** Handles input datasets, preprocessing, and feature engineering. The system uses Pandas for data manipulation and Scikit-learn for standard ML operations. Data is versioned and tracked alongside model artifacts.

**Training Layer:** Orchestrates model training with automated logging to MLflow. This layer integrates Optuna for hyperparameter optimization, enabling systematic exploration of the parameter space. Each training run is logged with parameters, metrics, and artifacts.

**Registry Layer:** Manages model lifecycle through MLflow's Model Registry. Models are versioned, tagged with metadata, and transitioned through stages (Staging, Production) with full audit trails.

### Data Flow

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Hyperparameter Optimization
    ↓                                                     ↓
  Logged                                          MLflow Tracking Server
                                                         ↓
                                          Metrics, Parameters, Artifacts
                                                         ↓
                                              Model Registry & Versioning
```

---

## Technical Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Experiment Tracking** | MLflow | 2.x | Central tracking server for all experiments |
| **Hyperparameter Optimization** | Optuna | 3.x | Automated parameter search and optimization |
| **Machine Learning** | Scikit-learn | 1.x | Model training and evaluation |
| **Data Processing** | Pandas | 2.x | Data manipulation and feature engineering |
| **Numerical Computing** | NumPy | 1.x | Array operations and calculations |
| **Metrics Visualization** | Matplotlib | 3.x | Local experiment visualization |
| **Environment Management** | Python venv | 3.11+ | Isolated Python environment |

---

## Project Structure

```
project1-mlflow/
├── data/
│   ├── raw/                    # Original, immutable datasets
│   ├── processed/              # Cleaned and feature-engineered data
│   └── splits/                 # Train/test/validation splits
├── models/
│   └── artifacts/              # Saved model binaries and metadata
├── src/
│   ├── train.py               # Main training orchestration script
│   ├── config.py              # Configuration and hyperparameters
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── evaluation.py          # Model evaluation metrics
│   └── utils.py               # Helper functions
├── notebooks/
│   └── exploration.ipynb      # Data exploration and analysis
├── mlruns/                     # MLflow local tracking directory
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Implementation Details

### 1. Environment Setup

**Installation Steps:**

```bash
# Clone the repository
git clone <repository-url>
cd project1-mlflow

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mlflow; print(mlflow.__version__)"
```

**Key Dependencies:**

The `requirements.txt` file specifies exact versions to ensure reproducibility across different environments. MLflow is pinned to version 2.x for compatibility with modern tracking APIs. Optuna is included for hyperparameter optimization, and Scikit-learn provides the ML algorithms.

### 2. Configuration Management

**Configuration File (`config.py`):**

```python
# Hyperparameter search space
PARAM_SPACE = {
    'n_estimators': (50, 300),
    'max_depth': (3, 15),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
    'learning_rate': (0.001, 0.1)
}

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Random Forest Optimization"
MLFLOW_ARTIFACT_PATH = "./mlruns"

# Training settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
N_TRIALS = 50  # Number of Optuna optimization trials
```

This centralized configuration approach enables easy modification of hyperparameters and tracking settings without editing the main training script.

### 3. Data Preprocessing

**Preprocessing Pipeline:**

The preprocessing module implements a reproducible data transformation pipeline:

- **Data Loading:** Reads raw datasets from the `data/raw/` directory with error handling for missing files.
- **Cleaning:** Handles missing values using mean imputation for numerical features and mode imputation for categorical features.
- **Feature Engineering:** Creates derived features such as polynomial features and interaction terms to improve model expressiveness.
- **Normalization:** Applies StandardScaler to ensure all features are on the same scale, which is critical for tree-based models and distance-based algorithms.
- **Splitting:** Divides data into training, validation, and test sets with stratification for classification tasks to maintain class distribution.

**Reproducibility:** All preprocessing steps use fixed random seeds to ensure identical results across multiple runs. This is essential for comparing model performance across experiments.

### 4. Model Training with MLflow

**Training Script (`train.py`):**

The main training orchestration script integrates MLflow tracking at every step:

```python
import mlflow
import mlflow.sklearn
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# Set MLflow tracking server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Start MLflow run
with mlflow.start_run(run_name="baseline_model"):
    # Log parameters
    mlflow.log_params({
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    })
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Log metrics
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    mlflow.log_metrics({
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("plots/confusion_matrix.png")
```

**Key Features:**

- **Automatic Logging:** MLflow automatically captures model parameters, metrics, and artifacts without manual intervention.
- **Run Tracking:** Each training run is assigned a unique ID, enabling easy comparison and reproduction.
- **Artifact Storage:** Model files, plots, and other outputs are stored alongside run metadata for complete reproducibility.

### 5. Hyperparameter Optimization with Optuna

**Optimization Process:**

Optuna is integrated to systematically search the hyperparameter space and identify optimal configurations:

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Return validation score
    return model.score(X_val, y_val)

# Create study and optimize
study = optuna.create_study(direction='maximize')
mlflow_callback = MLflowCallback(tracking_uri=MLFLOW_TRACKING_URI)
study.optimize(objective, n_trials=50, callbacks=[mlflow_callback])

# Log best parameters
best_trial = study.best_trial
mlflow.log_params(best_trial.params)
```

**Optimization Strategy:**

Optuna uses a tree-structured Parzen estimator (TPE) sampler by default, which balances exploration and exploitation. This approach efficiently navigates the hyperparameter space and converges to optimal values faster than grid search or random search.

### 6. Model Registry and Versioning

**Model Registration:**

Once training is complete, models are registered in MLflow's Model Registry for lifecycle management:

```python
# Register model
model_uri = "runs:/<run-id>/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="RandomForest-Classifier"
)

# Transition to Staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="RandomForest-Classifier",
    version=registered_model.version,
    stage="Staging"
)
```

**Lifecycle Stages:**

- **None:** Initial stage after registration, used for testing and validation.
- **Staging:** Model is validated and ready for pre-production testing.
- **Production:** Model is approved for serving to end users.
- **Archived:** Model is deprecated and no longer actively used.

This structured lifecycle ensures that only validated models reach production, reducing the risk of deploying suboptimal models.

### 7. Experiment Tracking and Comparison

**Accessing Experiments:**

MLflow provides a web UI for visualizing and comparing experiments:

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

The UI displays:

- **Experiment List:** All experiments organized by name with run counts and timestamps.
- **Run Details:** Parameters, metrics, artifacts, and system information for each run.
- **Comparison View:** Side-by-side comparison of multiple runs to identify the best-performing configuration.
- **Metrics Charts:** Time-series plots showing metric evolution across runs.

**Programmatic Access:**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Query experiments
experiments = client.search_experiments()

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.val_accuracy > 0.9"
)

# Get best run
best_run = max(runs, key=lambda r: r.data.metrics['val_accuracy'])
print(f"Best run: {best_run.info.run_id}")
print(f"Best accuracy: {best_run.data.metrics['val_accuracy']}")
```

---

## Execution Guide

### Step 1: Prepare Data

```bash
# Place raw data in data/raw/ directory
# The preprocessing script will automatically:
# - Load data
# - Clean missing values
# - Engineer features
# - Create train/test splits
```

### Step 2: Start MLflow Tracking Server

```bash
# Terminal 1: Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access UI at http://localhost:5000
```

### Step 3: Run Training with Optimization

```bash
# Terminal 2: Run training script
python src/train.py

# Expected output:
# [I 2026-01-25 10:30:00,123] A new study created in memory with name: no-name-xxxxx.
# [I 2026-01-25 10:30:05,456] Trial 0 finished with value: 0.92 and parameters: {...}
# [I 2026-01-25 10:30:10,789] Trial 1 finished with value: 0.94 and parameters: {...}
# ...
# Best trial: Trial 49 with value: 0.96
```

### Step 4: Monitor Experiments

- Open http://localhost:5000 in your browser
- Navigate to the experiment to view all runs
- Compare runs to identify the best-performing configuration
- Examine metrics charts and parameter distributions

### Step 5: Register Best Model

```bash
# The training script automatically registers the best model
# Verify in MLflow UI under Models section
```

---

## Key Metrics and Evaluation

The project tracks the following metrics for model evaluation:

| Metric | Definition | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness; use when classes are balanced |
| **Precision** | TP / (TP + FP) | Relevance of positive predictions; minimize false positives |
| **Recall** | TP / (TP + FN) | Coverage of positive class; minimize false negatives |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean; use when balancing precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across thresholds |

**Logging Metrics:**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1': f1_score(y_test, y_pred, average='weighted')
}

mlflow.log_metrics(metrics)
```

---

## Reproducibility and Best Practices

### Ensuring Reproducibility

**Random Seed Management:**

```python
import numpy as np
import random

RANDOM_STATE = 42

# Set seeds
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```

**Dependency Pinning:**

All dependencies are pinned to specific versions in `requirements.txt` to ensure identical behavior across different environments and time periods.

**Artifact Versioning:**

All model artifacts, including the trained model binary, preprocessing objects, and feature names, are stored with the MLflow run. This enables complete model reconstruction at any point in the future.

### Production Considerations

**Model Serving:**

Once a model is promoted to Production in the Model Registry, it can be served using MLflow Models:

```bash
mlflow models serve -m "models:/RandomForest-Classifier/Production" --port 8000
```

**Monitoring:**

In production, continuously monitor model performance against baseline metrics. If performance degrades, trigger retraining with new data.

**Versioning Strategy:**

Maintain a clear versioning strategy where each model version corresponds to a specific training run. This enables easy rollback if a new model performs worse than the previous version.

---

## Troubleshooting

### Common Issues

**Issue: MLflow Tracking Server Connection Error**

*Solution:* Ensure the MLflow UI is running on the correct host and port. Verify the `MLFLOW_TRACKING_URI` in `config.py` matches the server address.

**Issue: Out of Memory During Training**

*Solution:* Reduce the number of Optuna trials or decrease the dataset size. Consider using data sampling for initial experiments.

**Issue: Model Performance Degradation**

*Solution:* Check for data drift by comparing feature distributions between training and production data. Retrain the model with recent data if drift is detected.

---

## References

1. [MLflow Documentation - Tracking](https://mlflow.org/docs/latest/tracking.html)
2. [Optuna Documentation - Hyperparameter Optimization](https://optuna.readthedocs.io/)
3. [Scikit-learn Documentation - Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
4. [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## Next Steps

1. **Extend to Production:** Deploy the model using MLflow Models or containerize with Docker for scalable serving.
2. **Add Monitoring:** Implement model performance monitoring and data drift detection for production models.
3. **Integrate with CI/CD:** Automate model training and registration as part of a CI/CD pipeline (see Project 3).
4. **Scale with Kubernetes:** Deploy the MLflow tracking server and model serving endpoints on Kubernetes for enterprise scalability (see Project 2).
