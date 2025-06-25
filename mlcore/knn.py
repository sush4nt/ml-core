import numpy as np
from joblib import Parallel, delayed


class CustomKNN:
    def __init__(self, k=1, metric="euclidean", ord=None, n_jobs=-1):
        self.k = k
        self.metric = metric
        self.ord = ord
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.k > X.shape[0]:
            raise ValueError("k cannot be greater than the number of training samples.")
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_neighbors)(x) for x in np.array(X)
        )
        # y_preds = [self._get_neighbors(x) for x in X]
        return np.array(y_preds)

    def _get_neighbors(self, x):
        distances = self._get_distances(x)
        # distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        sorted_indices = np.argpartition(distances, self.k)[: self.k]
        nearest_labels = self.y_train[sorted_indices]
        return np.bincount(nearest_labels).argmax()

    def _get_distances(self, x):
        assert self.X_train is not None, "Model must be fitted before predicting."
        assert self.metric in [
            "euclidean",
            "manhattan",
            "minkowski",
        ], "Unsupported metric"
        if self.metric == "euclidean":
            return np.linalg.norm(self.X_train - x, axis=1)
        elif self.metric == "manhattan":
            return np.linalg.norm(self.X_train - x, ord=1, axis=1)
        elif self.metric == "minkowski":
            assert self.ord is not None, "ord must be specified for Minkowski distance"
            return np.linalg.norm(self.X_train - x, ord=self.ord, axis=1)
