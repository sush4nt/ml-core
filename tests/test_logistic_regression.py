import numpy as np
import pytest
from mlcore.logistic_regression import CustomLogisticRegression

@pytest.fixture
def simple_logistic_data():
    # a trivial separable dataset
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y

def test_logistic_regression_training(simple_logistic_data):
    X, y = simple_logistic_data
    model = CustomLogisticRegression(learning_rate=0.01, n_iters=5000)
    model.fit(X, y)
    preds = model.predict(X)
    # should perfectly separate
    assert np.array_equal(preds, y)

def test_predict_proba_range(simple_logistic_data):
    X, _ = simple_logistic_data
    model = CustomLogisticRegression()
    # without training, sigmoid(0) = 0.5 for all
    probs = model.predict_proba(X)
    assert np.allclose(probs, 0.5)
