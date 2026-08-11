__version__ = "0.1.0"

from .adaboost import CustomAdaBoostClassifier
from .decision_tree import CustomDecisionTreeClassifier, CustomDecisionTreeRegressor
from .gaussian_nb import CustomGaussianNB
from .gradientboost import CustomGradientBoostingClassifier
from .knn import CustomKNN
from .linear_regression import CustomLinearRegression
from .linear_svm import CustomLinearSVM
from .logistic_regression import CustomLogisticRegression
from .random_forest import CustomRandomForestClassifier

__all__ = [
    "CustomLinearRegression",
    "CustomLogisticRegression",
    "CustomKNN",
    "CustomDecisionTreeClassifier",
    "CustomDecisionTreeRegressor",
    "CustomRandomForestClassifier",
    "CustomAdaBoostClassifier",
    "CustomGradientBoostingClassifier",
    "CustomGaussianNB",
    "CustomLinearSVM",
]
# it will now help expose clean public API without needing to know internal module names. :)
