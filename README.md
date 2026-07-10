# GBRF — Gradient Boosted Random Forest

[![PyPI version](https://badge.fury.io/py/gbrf.svg)](https://pypi.org/project/gbrf/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A novel hybrid machine learning algorithm** combining Random Forest and Gradient Boosting. At each boosting iteration, a full Random Forest ensemble learns the pseudo-residuals, providing superior variance reduction compared to standard Gradient Boosting.

---

## What Makes GBRF Different?

| Algorithm | Variance Control | Bias Reduction | Overfitting Risk |
|-----------|---------------|----------------|------------------|
| Random Forest | :white_check_mark: High (bagging) | :x: Low | Low |
| Gradient Boosting | :x: Low | :white_check_mark: High | High |
| **GBRF (Ours)** | :white_check_mark: **High** (RF per iteration) | :white_check_mark: **High** (boosting) | **Moderate** |

**The Innovation:** Instead of a single decision tree learning residuals at each boosting step (standard GB), GBRF uses a **Random Forest ensemble** per iteration. This gives you:
- **Better generalization** than Gradient Boosting alone
- **Faster convergence** than Random Forest alone
- **Built-in early stopping** via out-of-bag monitoring
- **Aggregated feature importance** across all boosting rounds

---

## Installation

```bash
pip install hybridgbrf
```

### Requirements
- Python >= 3.8
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- joblib >= 1.1.0

---

## Quick Start

### Regression

```python
from hybridgbrf import GRFRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train GBRF
model = GRFRegressor(
    n_iterations=50,
    n_estimators_per_iteration=10,
    learning_rate=0.1,
    max_depth=3,
    early_stopping_rounds=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
print(f"R^2 Score: {model.score(X_test, y_test):.4f}")
print(f"MSE: {model.mse(X_test, y_test):.4f}")

# Feature importance
model.plot_feature_importances(top_n=10, save_path="importance.png")

# Training history
model.plot_training_history(save_path="history.png")
```

### Classification

```python
from hybridgbrf import GRFClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GRFClassifier(
    n_iterations=50,
    n_estimators_per_iteration=10,
    learning_rate=0.1,
    early_stopping_rounds=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(model.classification_report(X_test, y_test))
```

---

## API Reference

### `GRFRegressor`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iterations` | int | 100 | Boosting iterations (rounds) |
| `n_estimators_per_iteration` | int | 10 | Trees in each Random Forest |
| `learning_rate` | float | 0.1 | Shrinkage factor |
| `max_depth` | int | 3 | Max tree depth |
| `min_samples_split` | int | 2 | Min samples to split |
| `min_samples_leaf` | int | 1 | Min samples at leaf |
| `max_features` | str/int/float | 'sqrt' | Features per split |
| `subsample` | float | 0.8 | Sample fraction per iteration |
| `early_stopping_rounds` | int | 10 | Stop if no improvement |
| `validation_fraction` | float | 0.1 | Validation split size |
| `tol` | float | 1e-4 | Improvement threshold |
| `random_state` | int | None | Reproducibility seed |
| `verbose` | int | 0 | Verbosity (0-2) |
| `n_jobs` | int | -1 | Parallel jobs (-1 = all CPUs) |

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train the model |
| `predict(X)` | Predict targets |
| `score(X, y)` | R^2 score (regression) / Accuracy (classification) |
| `mse(X, y)` | Mean Squared Error |
| `mae(X, y)` | Mean Absolute Error |
| `cross_validate(X, y, cv)` | Cross-validation |
| `grid_search(X, y, param_grid)` | Hyperparameter tuning |
| `plot_feature_importances(...)` | Plot aggregated importances |
| `plot_training_history(...)` | Plot loss curves |
| `save_model(filepath)` | Save to disk |
| `load_model(filepath)` | Load from disk (class method) |

---

## How It Works

```
Standard Gradient Boosting:          GBRF (Our Hybrid):
+--------------------------------+   +--------------------------------+
|  Iteration 1                   |   |  Iteration 1                   |
|  Tree -> Residual Prediction   |   |  RF Ensemble -> Residuals      |
+--------------------------------+   +--------------------------------+
              |                                    |
              v                                    v
+--------------------------------+   +--------------------------------+
|  Iteration 2                   |   |  Iteration 2                   |
|  Tree -> Residual Prediction   |   |  RF Ensemble -> Residuals      |
+--------------------------------+   +--------------------------------+
              |                                    |
              v                                    v
       ... continues ...                    ... continues ...

Key Difference: Single Tree vs Random Forest per iteration
```

---

## Performance Comparison

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from hybridgbrf import GRFRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"RF R^2: {rf.score(X_test, y_test):.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
print(f"GB R^2: {gb.score(X_test, y_test):.4f}")

# GBRF (Ours)
hgbrf = GRFRegressor(n_iterations=50, n_estimators_per_iteration=10, random_state=42)
hgbrf.fit(X_train, y_train)
print(f"GBRF R^2: {hgbrf.score(X_test, y_test):.4f}")
```

---

## Saving and Loading

```python
# Save
model.save_model("my_gbrf_model.pkl")

# Load
from gbrf import GRFRegressor
model = GRFRegressor.load_model("my_gbrf_model.pkl")
```

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

**Built with :heart: by Abhishek Singh**
