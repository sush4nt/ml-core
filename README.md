# ml-core

From-scratch implementations of classical ML algorithms in Python — no scikit-learn under the hood.
Built for learning, experimentation, and understanding what happens inside the black box.

---

## Algorithms

| Algorithm | Type | Module |
|---|---|---|
| Linear Regression | Regression | `mlcore.linear_regression` |
| Logistic Regression | Classification | `mlcore.logistic_regression` |
| K-Nearest Neighbours | Classification | `mlcore.knn` |
| Decision Tree | Classification / Regression | `mlcore.decision_tree` |
| Random Forest | Classification | `mlcore.random_forest` |
| AdaBoost | Classification | `mlcore.adaboost` |
| Gradient Boosting | Classification | `mlcore.gradient_boost` |
| Gaussian Naive Bayes | Classification | `mlcore.gaussian_nb` |
| Linear SVM | Classification | `mlcore.linear_svm` |

---

## Project Structure

```
ml-core/
├── mlcore/          # algorithm implementations
├── notebooks/       # per-algorithm usage examples with real datasets
├── tests/           # unit tests
├── setup.py
└── requirements.txt
```

---

## Installation

**From PyPI:**
```bash
pip install mlcore
```

**From source:**
```bash
git clone https://github.com/sush4nt/ml-core
cd ml-core
pip install -e .
```

---

## Quick Start

```python
from mlcore import CustomLinearRegression, CustomKNN, CustomDecisionTreeClassifier

# Regression
reg = CustomLinearRegression()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)

# Classification
clf = CustomKNN(k=5)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# Gradient Boosting
gb = CustomGradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    loss="log_loss",
    max_depth=3,
    verbose=True,
)
gb.fit(X_train, y_train)
preds = gb.predict(X_test)
```

All classes follow a consistent `fit` / `predict` interface.
See `notebooks/` for full worked examples.

---

## Development Setup

```bash
pip install -r requirements.txt
pytest tests/
```

---

## Requirements

- Python >= 3.9
- numpy
- pandas

---

## Key points updated

1. Module path corrected: mlcore.gradientboost → mlcore.gradient_boost (the actual file is mlcore/gradient_boost.py).
2. Added `tqdm` to requirements in README and changelog.
3. Added a gradient boosting quick-start snippet.
4. Bumped version to `0.1.3` to match pyproject.toml#L7.
5. Documented max_thresholds and `verbose` additions.

One thing to verify: if `tqdm` is now a runtime dependency, it should also be added to pyproject.toml#L17 `dependencies` and requirements.txt. The README change above assumes you've done that; if not, add it to those files as well before cutting the release.

---