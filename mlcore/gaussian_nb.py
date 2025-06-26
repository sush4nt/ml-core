import numpy as np


class CustomGaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes, self.counts = np.unique(y, return_counts=True)
        self.n_classes = len(self.classes)
        self.prior_probs = self.counts / y.shape[0]
        print(self.classes, self.n_features, self.n_classes, self.prior_probs)

        self.mean = np.zeros((self.n_classes, self.n_features))
        self.var = np.zeros((self.n_classes, self.n_features))

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)

    def predict(self, X):
        predictions = np.zeros((X.shape[0],), dtype=int)

        for i, x in enumerate(X):
            posterior_probs = self._calculate_log_likelihood(x)
            class_i = np.argmax(posterior_probs)
            predictions[i] = self.classes[class_i]
        return predictions

    def _calculate_log_likelihood(self, x):
        log_probs = np.zeros((self.n_classes,))

        for i, c in enumerate(self.classes):
            prior = self.prior_probs[i]
            mean = self.mean[i]
            var = self.var[i]

            log_coef = -0.5 * (np.log(2 * np.pi * var))
            expo_term = -0.5 * ((x - mean) ** 2) / var
            log_probs[i] = np.log(prior) + np.sum(log_coef + expo_term)

        return log_probs
