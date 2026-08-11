import numpy as np

from mlcore.decision_tree import CustomDecisionTreeClassifier


class CustomGradientBoostingClassifier:
    pass


#     def __init__(
#         self,
#         n_estimators=50,
#         learning_rate=0.1,
#         max_depth=3,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_impurity_decrease=1e-7,
#         random_state=None,
#         **tree_kwargs
#     ):
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_impurity_decrease = min_impurity_decrease
#         self.random_state = random_state
#         self.rnd = np.random.RandomState(self.random_state)
#         self.root = None
#         self.feature_importances_ = None

#     def fit(self, X, y):
#         # convert to arrays
#         X = np.array(X)
#         y = np.array(y)
#         num_samples, num_features = X.shape
#         self.classes_ = np.unique(y)
#         self.estimators_ = []
#         self.f0_ =
