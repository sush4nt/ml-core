from collections import Counter

import numpy as np


class DecisionNode:
    def __init__(self, feature_index, threshold, gain, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right


class LeafNode:
    def __init__(self, prediction, samples_count):
        self.prediction = prediction
        self.samples_count = samples_count


class CustomDecisionTreeClassifier:
    def __init__(
        self,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=1e-7,
        criterion="gini",
        n_classes=None,
        random_state=42,
        max_thresholds=None
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.n_classes = n_classes
        self.max_thresholds = max_thresholds
        self.root = None
        self.rnd = np.random.RandomState(random_state)
        # history
        self.cost_history = []
        self.feature_importances_ = None

    def fit(self, X, y, sample_weights=None):
        X = np.array(X)
        y = np.array(y, dtype=int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))

        num_samples, num_features = X.shape

        if sample_weights is None:
            sample_weights = np.ones(num_samples) / num_samples
        else:
            # use normalized sample weights
            sample_weights = np.array(sample_weights)
            sample_weights /= np.sum(sample_weights)

        self.cost_history = []
        self.feature_importances_ = np.zeros(num_features)

        self.root = self._build_tree(X, y, sample_weights, depth=0)

        # normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def predict(self, X):
        """Traverses through the trained tree to make predictions."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build_tree(self, X, y, sample_weights, depth):
        """
        1. Define criterions
        2. Calculates initial impurity
        3. Finds the best feature and threshold to split on
        4. Calculates information gain
        5. Recursively builds left and right subtrees
        6. Returns a leaf node if stopping condition is met otherwise a DecisionNode is returned.
        """
        num_samples, num_features = X.shape
        # if condition met, assign it as a leaf node
        if (
            depth == self.max_depth
            or num_samples < self.min_samples_split
            or len(set(y)) == 1
        ):
            weighted_counts = np.bincount(
                y, weights=sample_weights, minlength=self.n_classes
            )
            leaf_label = int(np.argmax(weighted_counts))
            return LeafNode(leaf_label, samples_count=num_samples)

        parent_impurity = self._calc_impurity(y, sample_weights)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        # no of features to consider when looking for best split.
        feature_sample_size = self._max_features_sample(num_features)
        # randomly chooses `feature_sample_size` features from the total features without replacement
        selected_features = self._random_feature_subsample(
            num_features, feature_sample_size
        )
        # find best split
        for feat in selected_features:
            unique_vals = np.unique(X[:, feat])
            if unique_vals.shape[0] < 2:
                continue
            thresholds = self._candidate_thresholds(unique_vals)
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                gain = self._calc_information_gain(
                    y, left_mask, right_mask, parent_impurity, sample_weights
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        # if no valid split or insufficient gain
        if best_feat is None or best_gain < self.min_impurity_decrease:
            weighted_counts = np.bincount(
                y, weights=sample_weights, minlength=self.n_classes
            )
            leaf_label = int(np.argmax(weighted_counts))
            return LeafNode(leaf_label, samples_count=num_samples)

        weighted_gain = np.sum(sample_weights) * best_gain
        self.cost_history.append(weighted_gain)
        self.feature_importances_[best_feat] += weighted_gain

        # split on best, and recursively build left and right subtrees
        mask_left = X[:, best_feat] <= best_thresh
        mask_right = ~mask_left
        left_subtree = self._build_tree(
            X[mask_left], y[mask_left], sample_weights[mask_left], depth + 1
        )
        right_subtree = self._build_tree(
            X[mask_right], y[mask_right], sample_weights[mask_right], depth + 1
        )
        return DecisionNode(
            best_feat, best_thresh, best_gain, left_subtree, right_subtree
        )

    def _candidate_thresholds(self, unique_vals):
        """Quantile-capped split candidates, or all midpoints if uncapped/few values."""
        n_unique = unique_vals.shape[0]
        if self.max_thresholds is None or n_unique - 1 <= self.max_thresholds:
            return (unique_vals[:-1] + unique_vals[1:]) / 2
        quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
        return np.unique(np.quantile(unique_vals, quantiles))

    def _calc_information_gain(
        self, y, left_mask, right_mask, parent_impurity, sample_weights
    ):
        w_total = np.sum(sample_weights)
        w_left = np.sum(sample_weights[left_mask])
        w_right = np.sum(sample_weights[right_mask])
        imp_left = self._calc_impurity(y[left_mask], sample_weights[left_mask])
        imp_right = self._calc_impurity(y[right_mask], sample_weights[right_mask])
        child_impurity = (w_left / w_total) * imp_left + (w_right / w_total) * imp_right
        return parent_impurity - child_impurity

    def _calc_impurity(self, y, sample_weights):
        assert self.criterion in (
            "gini",
            "entropy",
            "misclassification",
        ), "criterion must be 'gini', 'entropy', or 'misclassification'"
        weighted_counts = np.bincount(
            y, weights=sample_weights, minlength=self.n_classes
        )
        ps = weighted_counts / weighted_counts.sum()
        if self.criterion == "gini":
            return 1 - np.sum(ps**2)
        elif self.criterion == "entropy":
            return -np.sum([p * np.log2(p) for p in ps if p > 0])
        else:  # misclassification
            return 1 - np.max(ps)

    def _predict_one(self, x, node):
        if isinstance(node, LeafNode):
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _random_feature_subsample(self, n_features, feature_sample_size):
        """Create random feature subsample without replacement"""
        return self.rnd.choice(n_features, size=feature_sample_size, replace=False)

    def _max_features_sample(self, n_features):
        assert self.max_features in ("sqrt", "log2") or isinstance(
            self.max_features, int
        ), "max_features must be 'sqrt', 'log2', or an integer"
        if isinstance(self.max_features, int):
            assert (
                self.max_features <= n_features
            ), "max_features must be less than or equal to the number of features in the dataset"
            assert (
                self.max_features > 1
            ), "max_features must be more than 1 to allow for splitting"
            return self.max_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))

    def print_tree(self):
        """Print tree with feature, threshold, gain, and sample counts."""
        self._print_node(self.root, spacing="")

    def _print_node(self, node, spacing):
        """Recursively prints the tree structure."""
        if isinstance(node, LeafNode):
            print(f"{spacing}Predict: {node.prediction} (samples={node.samples_count})")
            return
        print(
            f"{spacing}Feature[{node.feature_index}] ≤ {node.threshold:.4f}  |  Gain={node.gain:.4f}"
        )
        print(f"{spacing}→ True branch:")
        self._print_node(node.left, spacing + "    ")
        print(f"{spacing}→ False branch:")
        self._print_node(node.right, spacing + "    ")


class CustomDecisionTreeRegressor:
    def __init__(
        self,
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=1e-7,
        criterion="squared_error",
        random_state=42,
        max_thresholds=None,
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.root = None
        self.rnd = np.random.RandomState(random_state)
        self.max_thresholds = max_thresholds
        # history
        self.cost_history = []
        self.feature_importances_ = None

    def fit(self, X, y, sample_weights=None):
        X = np.array(X)
        y = np.array(y, dtype=float)

        num_samples, num_features = X.shape

        if sample_weights is None:
            sample_weights = np.ones(num_samples) / num_samples
        else:
            # use normalized sample weights
            sample_weights = np.array(sample_weights)
            sample_weights /= np.sum(sample_weights)

        self.cost_history = []
        self.feature_importances_ = np.zeros(num_features)

        self.root = self._build_tree(X, y, sample_weights, depth=0)

        # normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def predict(self, X):
        """Traverses through the trained tree to make predictions."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build_tree(self, X, y, sample_weights, depth):
        """
        1. Define criterions
        2. Calculates initial impurity
        3. Finds the best feature and threshold to split on
        4. Calculates information gain
        5. Recursively builds left and right subtrees
        6. Returns a leaf node if stopping condition is met otherwise a DecisionNode is returned.
        """
        num_samples, num_features = X.shape
        # if condition met, assign it as a leaf node
        if (
            depth == self.max_depth
            or num_samples < self.min_samples_split
            or np.average(
                (y - np.average(y, weights=sample_weights)) ** 2, weights=sample_weights
            )
            < 1e-10
        ):
            leaf_value = np.average(y, weights=sample_weights)
            return LeafNode(leaf_value, samples_count=num_samples)

        parent_impurity = self._calc_impurity(y, sample_weights)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        # no of features to consider when looking for best split.
        feature_sample_size = self._max_features_sample(num_features)
        # randomly chooses `feature_sample_size` features from the total features without replacement
        selected_features = self._random_feature_subsample(
            num_features, feature_sample_size
        )
        # find best split
        for feat in selected_features:
            unique_vals = np.unique(X[:, feat])
            if unique_vals.shape[0] < 2:
                continue
            thresholds = self._candidate_thresholds(unique_vals)
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                gain = self._calc_information_gain(
                    y, left_mask, right_mask, parent_impurity, sample_weights
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        # if no valid split or insufficient gain
        if best_feat is None or best_gain < self.min_impurity_decrease:
            leaf_value = np.average(y, weights=sample_weights)
            return LeafNode(leaf_value, samples_count=num_samples)

        weighted_gain = np.sum(sample_weights) * best_gain
        self.cost_history.append(weighted_gain)
        self.feature_importances_[best_feat] += weighted_gain

        # split on best, and recursively build left and right subtrees
        mask_left = X[:, best_feat] <= best_thresh
        mask_right = ~mask_left
        left_subtree = self._build_tree(
            X[mask_left], y[mask_left], sample_weights[mask_left], depth + 1
        )
        right_subtree = self._build_tree(
            X[mask_right], y[mask_right], sample_weights[mask_right], depth + 1
        )
        return DecisionNode(
            best_feat, best_thresh, best_gain, left_subtree, right_subtree
        )

    def _candidate_thresholds(self, unique_vals):
        """Quantile-capped split candidates, or all midpoints if uncapped/few values."""
        n_unique = unique_vals.shape[0]
        if self.max_thresholds is None or n_unique - 1 <= self.max_thresholds:
            return (unique_vals[:-1] + unique_vals[1:]) / 2
        quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
        return np.unique(np.quantile(unique_vals, quantiles))

    def _calc_information_gain(
        self, y, left_mask, right_mask, parent_impurity, sample_weights
    ):
        w_total = np.sum(sample_weights)
        w_left = np.sum(sample_weights[left_mask])
        w_right = np.sum(sample_weights[right_mask])
        imp_left = self._calc_impurity(y[left_mask], sample_weights[left_mask])
        imp_right = self._calc_impurity(y[right_mask], sample_weights[right_mask])
        child_impurity = (w_left / w_total) * imp_left + (w_right / w_total) * imp_right
        return parent_impurity - child_impurity

    def _calc_impurity(self, y, sample_weights):
        assert self.criterion in (
            "squared_error",
            "absolute_error",
            "poisson",
        ), "criterion must be 'squared_error', 'absolute_error', or 'poisson'"
        weighted_mean = np.average(y, weights=sample_weights)
        if self.criterion == "squared_error":
            return np.average((y - weighted_mean) ** 2, weights=sample_weights)
        elif self.criterion == "absolute_error":
            return np.average(np.abs(y - weighted_mean), weights=sample_weights)
        elif self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Poisson criterion requires non-negative target values."
                )
            safe_log = np.where(y > 0, np.log(y / weighted_mean + 1e-10), 0.0)
            return np.average(
                y * safe_log - (y - weighted_mean), weights=sample_weights
            )

    def _predict_one(self, x, node):
        if isinstance(node, LeafNode):
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _random_feature_subsample(self, n_features, feature_sample_size):
        """Create random feature subsample without replacement"""
        return self.rnd.choice(n_features, size=feature_sample_size, replace=False)

    def _max_features_sample(self, n_features):
        assert self.max_features in ("sqrt", "log2", None) or isinstance(
            self.max_features, int
        ), "max_features must be 'sqrt', 'log2', or an integer"
        if self.max_features==None:
            return n_features
        if isinstance(self.max_features, int):
            assert (
                self.max_features <= n_features
            ), "max_features must be less than or equal to the number of features in the dataset"
            assert (
                self.max_features >= 1
            ), "max_features must be more than or equal to 1 to allow for splitting"
            return self.max_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))

    def print_tree(self):
        """Print tree with feature, threshold, gain, and sample counts."""
        self._print_node(self.root, spacing="")

    def _print_node(self, node, spacing):
        """Recursively prints the tree structure."""
        if isinstance(node, LeafNode):
            print(f"{spacing}Predict: {node.prediction} (samples={node.samples_count})")
            return
        print(
            f"{spacing}Feature[{node.feature_index}] ≤ {node.threshold:.4f}  |  Gain={node.gain:.4f}"
        )
        print(f"{spacing}→ True branch:")
        self._print_node(node.left, spacing + "    ")
        print(f"{spacing}→ False branch:")
        self._print_node(node.right, spacing + "    ")
