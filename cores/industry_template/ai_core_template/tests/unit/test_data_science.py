import pandas as pd
import numpy as np
from aicore.pipelines.data_science.nodes import split_data, train_model

def test_split_data():
    df = pd.DataFrame({
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "target": np.random.rand(10)
    })
    parameters = {
        "features": ["feature1", "feature2"],
        "test_size": 0.2,
        "random_state": 42
    }
    X_train, X_test, y_train, y_test = split_data(df, parameters)
    assert len(X_train) == 8
    assert len(X_test) == 2

def test_train_model():
    X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y_train = pd.Series([1, 2, 3])
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")
