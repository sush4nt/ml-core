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
| Gradient Boosting | Classification | `mlcore.gradientboost` |
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