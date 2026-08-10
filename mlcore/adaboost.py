import numpy as np

from mlcore.decision_tree import CustomDecisionTreeClassifier


class CustomAdaBoostClassifier:
    def __init__(
        self, n_estimators=50, learning_rate=1.0, random_state=None, **stump_kwargs
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.stump_kwargs = {**stump_kwargs, "max_depth": 1}
        self.stump_learners = []
        self.rnd = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        # Convert to arrays and map classes
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        y_mapped = np.where(y == self.classes_[0], -1, 1)
        # Initialize with even sample weights
        self.sample_weights = np.full(num_samples, 1 / num_samples)
        # reset learner list when fitting
        self.stump_learners = []
        # tracking
        self.alphas_ = []
        self.cost_history_ = []
        self.feature_importances_ = np.zeros(num_features)

        for est in range(self.n_estimators):

            stump = CustomDecisionTreeClassifier(
                **self.stump_kwargs, random_state=self.rnd.randint(0, int(1e6))
            )
            stump.fit(X, y, sample_weights=self.sample_weights)
            stump_preds = stump.predict(X)
            mapped_stump_preds = np.where(stump_preds == self.classes_[0], -1, 1)

            # calculate weighted error
            misclassifications = mapped_stump_preds != y_mapped
            eps = np.dot(self.sample_weights, misclassifications)
            if eps >= 0.5 or eps <= 0:
                # stop boosting since misclf rate is too high
                break
            eps = np.clip(eps, 1e-10, 1 - 1e-10)

            # stump importance
            alpha = self.learning_rate * 0.5 * np.log((1 - eps) / eps)

            # reassign weights
            self.sample_weights = self.sample_weights * np.exp(
                -alpha * y_mapped * mapped_stump_preds
            )
            self.sample_weights /= np.sum(self.sample_weights)

            # store
            self.alphas_.append(alpha)
            self.cost_history_.append(eps)
            self.stump_learners.append((stump, alpha))

            # store feature importances
            self.feature_importances_ += alpha * stump.feature_importances_

        # normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        # normallize stump importances
        alpha_array = np.array(self.alphas_)
        total_alpha = np.sum(alpha_array)
        self.stump_importances_ = (
            alpha_array / total_alpha if total_alpha > 0 else alpha_array
        )

        return self

    def predict(self, X):
        X = np.array(X)
        stump_preds = np.zeros(X.shape[0])

        for stump, alpha in self.stump_learners:
            stump_preds += alpha * np.where(stump.predict(X) == self.classes_[0], -1, 1)

        return np.where(stump_preds >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        X = np.array(X)
        stump_preds = np.zeros(X.shape[0])

        for stump, alpha in self.stump_learners:
            stump_preds += alpha * np.where(stump.predict(X) == self.classes_[0], -1, 1)

        expF = np.exp(stump_preds)
        expmF = np.exp(-stump_preds)
        p_pos = expF / (expF + expmF)
        return np.vstack([1 - p_pos, p_pos]).T
