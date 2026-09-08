from tqdm import trange
import numpy as np

from mlcore.decision_tree import CustomDecisionTreeClassifier, CustomDecisionTreeRegressor


class CustomGradientBoostingClassifier:
    def __init__(
        self,
        loss="log_loss",
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=1e-7,
        random_state=None,
        verbose=False
    ):
        assert loss in ("log_loss", "exponential"), "loss must be 'log_loss' or 'exponential'"
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.rnd = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.feature_importances_ = None
        self.verbose = verbose

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        y_idx = np.array([np.where(self.classes_ == c)[0][0] for c in y])

        self.estimators_ = []
        self.cost_history_ = []
        self.feature_importances_ = np.zeros(num_features)

        # log_loss works in {0,1} space and exponential works in {-1,+1} space
        y_loss = y_idx.astype(float) if self.loss == "log_loss" else np.where(y_idx == 0, -1.0, 1.0)

        self.F0_ = self._initialize_predictions(y_loss)
        F = np.full(num_samples, self.F0_)

        for _ in trange(self.n_estimators, desc="Boosting", disable=not self.verbose):
            pseudo_residuals = self._negative_gradient(y_loss, F)

            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.rnd.randint(0, int(1e6)),
            )
            tree.fit(X, pseudo_residuals)
            F += self.learning_rate * tree.predict(X)

            self.cost_history_.append(self._compute_loss(y_loss, F))
            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _initialize_predictions(self, y):
        """Optimal constant F0 that minimises the chosen loss."""
        p0 = np.clip((y > 0).mean(), 1e-10, 1 - 1e-10)
        if self.loss == "log_loss":
            return np.log(p0 / (1 - p0))          # log-odds
        else:
            return 0.5 * np.log(p0 / (1 - p0))    # half of log-odds for {-1,+1}

    def _negative_gradient(self, y, F):
        """Pseudo-residuals: -dL/dF for the chosen loss."""
        if self.loss == "log_loss":
            return y - self._sigmoid(F)
        else:  # exponential: L = mean(exp(-y*F)), -dL/dF = y * exp(-y*F)
            return y * np.exp(-np.clip(y * F, -500, 500))

    def _compute_loss(self, y, F):
        if self.loss == "log_loss":
            p = np.clip(self._sigmoid(F), 1e-10, 1 - 1e-10)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        else:  # exponential
            return float(np.mean(np.exp(-np.clip(y * F, -500, 500))))

    def predict_proba(self, X):
        X = np.array(X)
        F = np.full(X.shape[0], self.F0_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        # log_loss: F = log-odds -> sigmoid(F); exponential: F = 0.5*log-odds -> sigmoid(2F)
        p = self._sigmoid(F) if self.loss == "log_loss" else self._sigmoid(2 * F)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))