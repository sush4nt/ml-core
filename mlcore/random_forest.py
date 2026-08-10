import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from mlcore.decision_tree import CustomDecisionTreeClassifier


class CustomRandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=1e-7,
        criterion="gini",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.n_classes = None
        # tracking
        self.feature_importances_ = None
        self.cost_history_per_tree_ = None
        self.mean_cost_reduction_ = None

    def fit(self, X, y):
        rnd = np.random.RandomState(self.random_state)
        X = np.array(X)
        y = np.array(y)
        self.n_classes = len(np.unique(y))
        n_samples, n_features = X.shape

        def build_and_fit_tree(seed):
            X_sample, y_sample = self._bootstrap_dataset(X, y, seed)
            tree = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                criterion=self.criterion,
                n_classes=self.n_classes,
                random_state=seed,
            )
            tree.fit(X_sample, y_sample)
            return tree

        seeds = [rnd.randint(0, int(1e6)) for _ in range(self.n_estimators)]
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(build_and_fit_tree)(seed) for seed in seeds
        )
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.trees], axis=0
        )
        self.cost_history_per_tree_ = [tree.cost_history for tree in self.trees]
        self.mean_cost_reduction_ = float(
            np.mean([sum(h) for h in self.cost_history_per_tree_])
        )
        return self

    def predict(self, X):
        X = np.array(X)
        # X: (m,n) -> Predictions: (m,)
        # (n_estimators, m) -> Transpose -> (m, n_estimators)
        # parallelize prediction across trees
        tree_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        tree_predictions = np.array(tree_predictions).T
        # perform majority voting
        predictions, _ = stats.mode(tree_predictions, axis=1, keepdims=False)
        return np.array(predictions)

    def _bootstrap_dataset(self, X, y, seed):
        rnd = np.random.RandomState(seed)
        n_samples = X.shape[0]
        indices = rnd.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
