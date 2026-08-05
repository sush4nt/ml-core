import numpy as np


class CustomLinearSVM:
    def __init__(
        self,
        learning_rate=0.01,
        C=0.1,
        n_iters=1000,
        l1=False,
        l2=False,
        alpha=1,
        tol=1e-4,
        patience=50,
    ):
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.tol = tol
        self.patience = patience

        # tracking history
        self.cost_history = []
        self.margin_cost_history = []
        self.misclassification_cost_history = []
        self.regularization_history = []

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        # (n,1)
        self.weights = np.zeros(n_cols)
        self.bias = 0

        no_improve_count = 0
        best_cost = np.inf
        for i in range(self.n_iters):
            # (m,n) x (n, 1) = (m, 1)
            # input vector projections on w vector
            # >0 means positive class, <0 means negative class
            projections = np.dot(X, self.weights) + self.bias
            margins = 1 - y * projections

            # cost function
            cost_margins = np.dot(self.weights, self.weights.T) / 2
            # hinge loss term for misclassified points
            cost_misclassifications = self.C * np.sum(np.maximum(0, margins))

            if self.l1:
                regularization_term = self.alpha * np.sum(np.abs(self.weights))
            elif self.l2:
                regularization_term = self.alpha * np.sum(np.square(self.weights))
            else:
                regularization_term = 0

            cost = cost_margins + cost_misclassifications + regularization_term

            # track history
            self.cost_history.append(cost)
            self.margin_cost_history.append(cost_margins)
            self.misclassification_cost_history.append(cost_misclassifications)
            self.regularization_history.append(regularization_term)

            # gradients
            # dW = self.weights - self.C * np.dot(X.T, y * (margins > 0)) / n_rows
            common_dW = self.weights - self.C * np.dot(X.T, y * (margins > 0)) / n_rows
            if self.l1:
                dW = common_dW + self.alpha / n_rows * np.sign(self.weights)
            elif self.l2:
                dW = common_dW + self.alpha / n_rows * self.weights
            else:
                dW = common_dW
            dB = 0 - self.C * np.sum(y * (margins > 0)) / n_rows

            # update weights and bias
            lr_decayed = self.learning_rate / (
                1 + self.learning_rate * (i + 1) / self.C
            )
            self.weights -= lr_decayed * dW
            self.bias -= lr_decayed * dB

            # early stopping: check if improvement is below tolerance
            if best_cost - cost < self.tol:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(
                        f"Early stopping at iteration {i} (no improvement for {self.patience} rounds)"
                    )
                    break
            else:
                best_cost = cost
                no_improve_count = 0

        return {
            "cost_history": self.cost_history,
            "margin_cost_history": self.margin_cost_history,
            "misclassification_cost_history": self.misclassification_cost_history,
            "regularization_history": self.regularization_history,
        }

    def predict(self, X):
        # (m,n) * (n,1) = (m,1)
        projections = np.dot(X, self.weights) + self.bias
        return np.sign(projections)
