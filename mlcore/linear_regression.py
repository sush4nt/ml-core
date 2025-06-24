import numpy as np


class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=100, l1=False, l2=False, alpha=1):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        self.weights = np.zeros(n_cols)
        self.bias = 0

        for _ in range(self.n_iters):
            # (m,n) x (n,1) = (m,1)
            hypothesis = np.dot(X, self.weights) + self.bias
            cost_mse = (1 / 2 * n_rows) * np.sum(np.square((hypothesis - y)))
            # apply regularization and calculate gradients
            if self.l1:
                regularization_term = self.alpha * np.sum(np.abs(self.weights))
            elif self.l2:
                regularization_term = self.alpha * np.sum(np.square(self.weights))
            else:
                regularization_term = 0
            # cost
            cost = cost_mse + regularization_term
            common_dw = (1 / n_rows) * np.dot(X.T, (hypothesis - y))
            dW = (
                (common_dw + self.alpha / n_rows * self.alpha * np.sign(self.weights))
                if self.l1
                else (
                    (common_dw + self.alpha / n_rows * self.weights)
                    if self.l2
                    else (common_dw)
                )
            )
            dB = (1 / n_rows) * np.sum(hypothesis - y)
            # update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

    def predict(self, X):
        # (m,n) x (n,1) = (m,1)
        return np.dot(X, self.weights) + self.bias
