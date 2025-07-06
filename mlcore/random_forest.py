import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from mlcore.decision_tree import CustomDecisionTreeClassifier

class CustomRandomForestClassifier():
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
        self.rnd = np.random.RandomState(self.random_state)

    def fit(self, X, y):
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
                random_state=seed
            )
            tree.fit(X_sample, y_sample)
            return tree
        
        seeds = [self.rnd.randint(0, int(1e6)) for _ in range(self.n_estimators)]
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(build_and_fit_tree)(seed) for seed in seeds
        )

    def predict(self, X):
        # X: (m,n) -> Predictions: (m,)
        # (n_estimators, m) -> Transpose -> (m, n_estimators)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees]).T
        # perform majority voting
        predictions = [Counter(p).most_common(1)[0][0] for p in tree_predictions]
        return np.array(predictions)

    def _bootstrap_dataset(self, X, y, seed):
        rnd = np.random.RandomState(seed)
        n_samples = X.shape[0]
        indices = rnd.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _max_features_sample(self, n_features):
        assert self.max_features in ("sqrt", "log2") or isinstance(self.max_features, int), \
            "max_features must be 'sqrt', 'log2', or an integer"
        if isinstance(self.max_features, int):
            assert self.max_features <= n_features, \
            "max_features must be less than or equal to the number of features in the dataset"
            return self.max_features
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        return n_features