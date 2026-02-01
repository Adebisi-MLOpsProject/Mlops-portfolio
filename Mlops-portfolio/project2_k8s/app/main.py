from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from prometheus_client import Counter, Histogram, make_asgi_app
import time

app = FastAPI(title="ML Model Serving API")

# Metrics
PREDICTION_COUNTER = Counter("model_predictions_total", "Total number of predictions")
LATENCY_HISTOGRAM = Histogram("model_prediction_latency_seconds", "Latency of predictions")

# Load a dummy model (in real case, load from MLflow)
data = load_diabetes()
model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model.fit(data.data, data.target)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: list):
    start_time = time.time()
    PREDICTION_COUNTER.inc()
    
    # Simple prediction logic
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    
    latency = time.time() - start_time
    LATENCY_HISTOGRAM.observe(latency)
    
    return {"prediction": prediction.tolist(), "latency": latency}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
