import numpy as np


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, l1=False, l2=False, alpha=1, tol=1e-4, patience=50):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.weights = None
        self.bias = None
        # for tracking
        self.cost_history = []
        self.logloss_history = []
        self.regularization_history = []
        self.tol = tol 
        self.patience = patience

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        self.weights = np.zeros(n_cols)
        self.bias = 0

        no_improve_count = 0
        best_cost = np.inf
        for iteration in range(self.n_iters):
            # (m,n) x (n,1) = (m,1)
            hypothesis = self.sigmoid(np.dot(X, self.weights) + self.bias)
            # logloss term
            cost_logloss = -(1 / n_rows) * np.sum(
                y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)
            )
            # find regularization cost
            if self.l1:
                regularization_term = (self.alpha / n_rows) * np.sum(
                    np.abs(self.weights)
                )
            elif self.l2:
                regularization_term = (self.alpha / (2 * n_rows)) * np.sum(
                    np.square(self.weights)
                )
            else:
                regularization_term = 0
            # calculate total cost
            cost = cost_logloss + regularization_term

            # tracking
            self.cost_history.append(cost)
            self.logloss_history.append(cost_logloss)
            self.regularization_history.append(regularization_term)

            # calculate gradients
            # (m, 1)
            dZ = hypothesis - y
            # (m,n) x (m,1) = (m,1)
            common_dw = (1 / n_rows) * np.dot(X.T, dZ)
            if self.l1:
                dW = (common_dw + self.alpha / n_rows * np.sign(self.weights))
            elif self.l2:
                dW = (common_dw + self.alpha / n_rows * self.weights)
            else:
                dW = common_dw
            # scalar
            dB = (1 / n_rows) * np.sum(dZ)
            # update weights
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            # early stopping: check if improvement is below tolerance
            if best_cost - cost < self.tol:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(f"Early stopping at iteration {iteration} (no improvement for {self.patience} rounds)")
                    break
            else:
                best_cost = cost
                no_improve_count = 0
        return {
            "cost_history": self.cost_history,
            "logloss_history": self.logloss_history,
            "regularization_history": self.regularization_history,
        }


    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            _, n_cols = X.shape
            self.weights = np.zeros(n_cols)
            self.bias = 0
        # (m,n) x (n,1) = (m,1)
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
