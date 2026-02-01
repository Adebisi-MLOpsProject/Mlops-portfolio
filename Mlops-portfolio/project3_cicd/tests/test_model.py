import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

def test_model_prediction():
    data = load_diabetes()
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(data.data, data.target)
    
    sample_input = data.data[0].reshape(1, -1)
    prediction = model.predict(sample_input)
    
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == 1
    assert prediction[0] > 0

def test_data_shape():
    data = load_diabetes()
    assert data.data.shape[1] == 10
