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
        min_samples_split=2,
        min_impurity_decrease=1e-7,
        criterion="gini",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.n_classes = None
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_classes = len(set(y))
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """Traverses through the trained tree to make predictions."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
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
            leaf_label = Counter(y).most_common(1)[0][0]
            return LeafNode(leaf_label, samples_count=num_samples)

        parent_impurity = self._calc_impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # find best split
        for feat in range(num_features):
            unique_vals = np.unique(X[:, feat])
            if unique_vals.shape[0] < 2:
                continue
            # use feature values' midpoints
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = self._calc_information_gain(
                    y, left_mask, right_mask, parent_impurity
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        # if no valid split or insufficient gain
        if best_feat is None or best_gain < self.min_impurity_decrease:
            leaf_label = Counter(y).most_common(1)[0][0]
            return LeafNode(leaf_label, samples_count=num_samples)

        # split on best, and recursively build left and right subtrees
        mask_left = X[:, best_feat] <= best_thresh
        mask_right = ~mask_left
        left_subtree = self._build_tree(X[mask_left], y[mask_left], depth + 1)
        right_subtree = self._build_tree(X[mask_right], y[mask_right], depth + 1)
        return DecisionNode(
            best_feat, best_thresh, best_gain, left_subtree, right_subtree
        )

    def _calc_information_gain(self, y, left_mask, right_mask, parent_impurity):
        n = len(y)
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        imp_left = self._calc_impurity(y[left_mask])
        imp_right = self._calc_impurity(y[right_mask])
        child_impurity = (n_left / n) * imp_left + (n_right / n) * imp_right
        return parent_impurity - child_impurity

    def _calc_impurity(self, y):
        assert self.criterion in (
            "gini",
            "entropy",
            "misclassification",
        ), "criterion must be 'gini', 'entropy', or 'misclassification'"
        counts = np.bincount(y, minlength=self.n_classes)
        ps = counts / counts.sum()
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
