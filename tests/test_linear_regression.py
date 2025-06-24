import numpy as np
import pytest
from mlcore.linear_regression import CustomLinearRegression

@pytest.fixture
def simple_linear_data():
    # y = 2*x + 1
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = 2 * X.flatten() + 1
    return X, y

def test_linear_regression_no_regularization(simple_linear_data):
    X, y = simple_linear_data
    model = CustomLinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    # weights ≈ 2, bias ≈ 1
    assert pytest.approx(model.weights[0], rel=1e-2) == 2.0
    assert pytest.approx(model.bias, rel=1e-2) == 1.0

def test_linear_predict(simple_linear_data):
    X, y = simple_linear_data
    model = CustomLinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-1)
