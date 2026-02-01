import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import optuna

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(n_estimators, max_depth):
    # Load dataset
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(nested=True):
        # Define model
        lr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        lr.fit(X_train, y_train)

        # Predict
        predicted_qualities = lr.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log model
        mlflow.sklearn.log_model(lr, "model")
        
        return rmse

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    return train(n_estimators, max_depth)

if __name__ == "__main__":
    mlflow.set_experiment("Diabetes_Regression_Project")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
