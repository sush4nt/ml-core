import numpy as np


class CustomLinearSVM:
    def __init__(self, learning_rate=0.01, C=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        # (n,1)
        self.weights = np.zeros(n_cols)
        self.bias = 0

        for i in range(self.n_iters):
            # (m,n) x (n, 1) = (m, 1)
            # input vector projections on w vector
            # >0 means positive class, <0 means negative class
            projections = np.dot(X, self.weights) + self.bias
            margins = 1 - y * projections

            # cost function
            cost_margins = np.dot(self.weights, self.weights.T) / 2
            cost_misclassifications = self.C * np.sum(np.maximum(0, margins))
            cost_margins + cost_misclassifications

            # gradients
            dW = self.weights - self.C * np.dot(X.T, y * (margins > 0))
            dB = 0 - self.C * np.sum(y * (margins > 0))

            # update weights and bias
            lr_decayed = self.learning_rate / (
                1 + self.learning_rate * (i + 1) / self.C
            )
            self.weights -= lr_decayed * dW
            self.bias -= lr_decayed * dB

    def predict(self, X):
        # (m,n) * (n,1) = (m,1)
        projections = np.dot(X, self.weights) + self.bias
        return np.sign(projections)
