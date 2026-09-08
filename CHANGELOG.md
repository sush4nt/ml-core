# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-08-11

### Added
- Initial release of mlcore
- Custom implementations of supervised machine learning algorithms
- Support for Python 3.9+
- Algorithms included: Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boost, AdaBoost, KNN, Gaussian Naive Bayes, Linear SVM

## [0.1.1] - 2026-08-11

### Added
- Testing and implemented version CI

## [0.1.2] - 2026-08-11

### Added
- Simple README documentation changes

## [0.1.3] - 2026-09-08

### Added
- `CustomGradientBoostingClassifier` implementation with `log_loss` and `exponential` loss functions
- `max_thresholds` option in `CustomDecisionTreeClassifier` and `CustomDecisionTreeRegressor` for quantile-capped split candidates, speeding up tree building
- Optional `tqdm` progress bar in `CustomGradientBoostingClassifier` via `verbose=True`
- `verbose` flag in `CustomGradientBoostingClassifier` to toggle progress display

### Changed
- `CustomGradientBoostingClassifier` module path corrected from `mlcore.gradientboost` to `mlcore.gradient_boost`
- Decision tree regressor default `max_features` changed to `None` (all features) for better boosting performance