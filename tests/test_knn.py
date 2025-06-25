import numpy as np
import pytest
from mlcore.knn import CustomKNN


@pytest.fixture
def simple_knn_data():
    # A 1D, trivially separable dataset:
    # points 0 and 1 → class 0; points 2 and 3 → class 1
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y


def test_k_greater_than_samples_raises(simple_knn_data):
    X, y = simple_knn_data
    model = CustomKNN(k=10)
    with pytest.raises(ValueError, match="k cannot be greater than"):
        model.fit(X, y)


def test_predict_euclidean_simple(simple_knn_data):
    X, y = simple_knn_data
    model = CustomKNN(k=1, metric="euclidean")
    model.fit(X, y)
    preds = model.predict(X)
    # each point should vote for itself
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    assert np.array_equal(preds, y)


def test_predict_manhattan(simple_knn_data):
    X, y = simple_knn_data
    # Manhattan should behave like L1; with k=1, same result
    model = CustomKNN(k=1, metric="manhattan")
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_predict_minkowski_ord3(simple_knn_data):
    X, y = simple_knn_data
    # Minkowski with ord=2 is identical to Euclidean
    model = CustomKNN(k=1, metric="minkowski", ord=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_minkowski_without_ord_raises(simple_knn_data):
    X, y = simple_knn_data
    model = CustomKNN(k=1, metric="minkowski", ord=None)
    model.fit(X, y)
    with pytest.raises(AssertionError, match="ord must be specified"):
        # distance computation should fail
        model.predict(X)


def test_unsupported_metric_raises(simple_knn_data):
    X, y = simple_knn_data
    model = CustomKNN(k=1, metric="chebyshev")
    model.fit(X, y)
    with pytest.raises(AssertionError, match="Unsupported metric"):
        model.predict(X)


def test_multi_dimensional_input():
    # 2D points: class 0 near (0,0), class 1 near (10,10)
    X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    y = np.array([0, 0, 1, 1])
    model = CustomKNN(k=1)
    model.fit(X, y)
    test_points = np.array([[0.2, -0.1], [10.5, 9.8]])
    preds = model.predict(test_points)
    assert np.array_equal(preds, [0, 1])
