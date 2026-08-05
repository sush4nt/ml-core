import numpy as np


class CustomLinearRegression:
    def __init__(
        self,
        learning_rate=0.01,
        n_iters=100,
        l1=False,
        l2=False,
        alpha=1,
        tol=1e-4,
        patience=50,
    ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.weights = None
        self.bias = None
        # for tracking
        self.cost_history = []
        self.mse_history = []
        self.regularization_history = []
        self.tol = tol
        self.patience = patience

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        self.weights = np.zeros(n_cols)
        self.bias = 0

        # reset
        self.cost_history = []
        self.mse_history = []
        self.regularization_history = []

        no_improve_count = 0
        best_cost = np.inf
        for iteration in range(self.n_iters):
            # (m,n) x (n,1) = (m,1)
            hypothesis = np.dot(X, self.weights) + self.bias
            cost_mse = (1 / (2 * n_rows)) * np.sum(np.square((hypothesis - y)))
            # apply regularization and calculate gradients
            if self.l1:
                regularization_term = self.alpha * np.sum(np.abs(self.weights))
            elif self.l2:
                regularization_term = self.alpha * np.sum(np.square(self.weights))
            else:
                regularization_term = 0
            # cost
            cost = cost_mse + regularization_term

            # tracking
            self.cost_history.append(cost)
            self.mse_history.append(cost_mse)
            self.regularization_history.append(regularization_term)

            common_dw = (1 / n_rows) * np.dot(X.T, (hypothesis - y))
            if self.l1:
                dW = common_dw + self.alpha / n_rows * np.sign(self.weights)
            elif self.l2:
                dW = common_dw + 2 * self.alpha / n_rows * self.weights
            else:
                dW = common_dw
            dB = (1 / n_rows) * np.sum(hypothesis - y)
            # update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            # early stopping: check if improvement is below tolerance
            if best_cost - cost < self.tol:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(
                        f"Early stopping at iteration {iteration} (no improvement for {self.patience} rounds)"
                    )
                    break
            else:
                best_cost = cost
                no_improve_count = 0
        return {
            "cost_history": self.cost_history,
            "mse_history": self.mse_history,
            "regularization_history": self.regularization_history,
        }

    def predict(self, X):
        # (m,n) x (n,1) = (m,1)
        return np.dot(X, self.weights) + self.bias
