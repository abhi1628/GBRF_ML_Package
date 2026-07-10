"""
GBRF - Gradient Boosted Random Forest

A novel hybrid algorithm that combines:
- Random Forest (bagging + feature randomness for variance reduction)
- Gradient Boosting (sequential residual learning for bias reduction)

At each boosting iteration, instead of a single decision tree learning residuals,
a full Random Forest ensemble learns the residuals. This provides:
1. Better variance control than standard Gradient Boosting
2. Better bias reduction than standard Random Forest
3. Natural feature importance aggregation across boosting rounds
4. Built-in early stopping via out-of-bag error monitoring

Author: Abhishek Singh
Version: 2.0.0
"""

import warnings
import logging
from typing import Union, Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _BaseGBRF(BaseEstimator, ABC):
    """
    Base class for Gradient Boosted Random Forest.
    
    This implements the core hybrid algorithm where each boosting iteration
    uses a Random Forest ensemble to learn the pseudo-residuals.
    """
    
    def __init__(
        self,
        n_estimators_per_iteration: int = 10,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 'sqrt',
        subsample: float = 0.8,
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 10,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        verbose: int = 0,
        n_jobs: int = -1
    ):
        """
        Initialize GBRF with hybrid parameters.
        
        Parameters
        ----------
        n_estimators_per_iteration : int, default=10
            Number of trees in the Random Forest at each boosting iteration.
            Higher = more variance reduction per iteration.
            
        n_iterations : int, default=100
            Number of boosting iterations (gradient boosting rounds).
            Each round adds a new Random Forest to the ensemble.
            
        learning_rate : float, default=0.1
            Shrinkage factor for each boosting iteration.
            Lower = slower learning, better generalization.
            
        max_depth : int, default=3
            Maximum depth of individual trees in the Random Forests.
            Controls complexity of base learners.
            
        min_samples_split : int, default=2
            Minimum samples required to split an internal node.
            
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node.
            
        max_features : {'sqrt', 'log2', int, float}, default='sqrt'
            Number of features to consider for best split.
            'sqrt' = sqrt(n_features), 'log2' = log2(n_features)
            
        subsample : float, default=0.8
            Fraction of samples used for fitting each Random Forest.
            Adds stochastic gradient boosting behavior.
            
        random_state : int, optional
            Random seed for reproducibility.
            
        early_stopping_rounds : int, optional, default=10
            Stop if validation score doesn't improve for this many iterations.
            None = disable early stopping.
            
        validation_fraction : float, default=0.1
            Fraction of training data to use for validation (early stopping).
            
        tol : float, default=1e-4
            Tolerance for early stopping improvement threshold.
            
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress, 2=detailed).
            
        n_jobs : int, default=-1
            Number of parallel jobs. -1 = all CPUs.
        """
        self.n_estimators_per_iteration = n_estimators_per_iteration
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        # Attributes set during fit
        self.estimators_ = []  # List of Random Forests, one per iteration
        self.initial_prediction_ = None
        self.feature_importances_ = None
        self.oob_scores_ = []
        self.train_scores_ = []
        self.val_scores_ = []
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        
    def _get_rf_params(self) -> Dict[str, Any]:
        """Get parameters for underlying Random Forest."""
        return {
            'n_estimators': self.n_estimators_per_iteration,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'oob_score': True,  # Enable for monitoring
            'bootstrap': True,
        }
    
    def _get_tree_params(self) -> Dict[str, Any]:
        """Get parameters for single tree (fallback)."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
        }
    
    def _validate_input(self, X, y=None, fit=False):
        """Validate input data."""
        if fit:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
            self.n_features_in_ = X.shape[1]
            return X, y
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} features, got {X.shape[1]}"
                )
            return X
    
    @abstractmethod
    def _compute_pseudo_residuals(self, y, predictions):
        """Compute pseudo-residuals for gradient boosting."""
        pass
    
    @abstractmethod
    def _get_base_estimator(self):
        """Get the base estimator (RF Regressor or Classifier)."""
        pass
    
    @abstractmethod
    def _get_initial_prediction(self, y):
        """Get initial prediction (mean for regression, log-odds for classification)."""
        pass
    
    @abstractmethod
    def _update_predictions(self, current_pred, rf_pred):
        """Update predictions with new Random Forest output."""
        pass
    
    @abstractmethod
    def _compute_loss(self, y_true, y_pred):
        """Compute loss for monitoring."""
        pass
    
    def fit(self, X, y):
        """
        Fit the Gradient Boosted Random Forest.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
        """
        # Validate inputs
        X, y = self._validate_input(X, y, fit=True)
        
        # Store feature names if DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            X = np.array(X)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split for early stopping validation
        if self.early_stopping_rounds is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Initial prediction
        self.initial_prediction_ = self._get_initial_prediction(y_train)
        current_pred_train = np.full(len(y_train), self.initial_prediction_)
        
        if X_val is not None:
            current_pred_val = np.full(len(y_val), self.initial_prediction_)
        
        # Boosting iterations
        best_val_score = float('inf') if hasattr(self, '_is_regressor') else -float('inf')
        rounds_without_improvement = 0
        
        for iteration in range(self.n_iterations):
            # Compute pseudo-residuals
            residuals = self._compute_pseudo_residuals(y_train, current_pred_train)
            
            # Sample for stochastic gradient boosting
            if self.subsample < 1.0:
                n_samples = int(len(X_train) * self.subsample)
                rng = np.random.RandomState(self.random_state + iteration)
                indices = rng.choice(len(X_train), n_samples, replace=False)
                X_sample = X_train[indices]
                res_sample = residuals[indices]
            else:
                X_sample = X_train
                res_sample = residuals
            
            # Fit Random Forest on residuals (THE HYBRID PART!)
            rf = self._get_base_estimator()
            rf.set_params(**self._get_rf_params())
            rf.fit(X_sample, res_sample)
            
            # Get predictions from this Random Forest
            rf_pred_train = rf.predict(X_train)
            
            # Update predictions
            current_pred_train = self._update_predictions(
                current_pred_train, rf_pred_train
            )
            
            # Store the Random Forest for this iteration
            self.estimators_.append(rf)
            
            # Compute training score
            train_loss = self._compute_loss(y_train, current_pred_train)
            self.train_scores_.append(train_loss)
            
            # Validation and early stopping
            if X_val is not None:
                rf_pred_val = rf.predict(X_val)
                current_pred_val = self._update_predictions(
                    current_pred_val, rf_pred_val
                )
                val_loss = self._compute_loss(y_val, current_pred_val)
                self.val_scores_.append(val_loss)
                
                # Check for improvement
                is_better = (
                    val_loss < (best_val_score - self.tol) 
                    if hasattr(self, '_is_regressor') 
                    else val_loss > (best_val_score + self.tol)
                )
                
                if is_better:
                    best_val_score = val_loss
                    rounds_without_improvement = 0
                    self.best_iteration_ = iteration
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose >= 1:
                        logger.info(
                            f"Early stopping at iteration {iteration + 1}. "
                            f"Best iteration: {self.best_iteration_ + 1}"
                        )
                    break
            
            if self.verbose >= 2 and (iteration + 1) % 10 == 0:
                msg = f"Iteration {iteration + 1}/{self.n_iterations} - Train Loss: {train_loss:.6f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        # Trim estimators to best iteration if early stopping
        if hasattr(self, 'best_iteration_'):
            self.estimators_ = self.estimators_[:self.best_iteration_ + 1]
        
        # Compute aggregated feature importances
        self._compute_feature_importances()
        
        self.is_fitted_ = True
        return self
    
    def _compute_feature_importances(self):
        """Aggregate feature importances across all boosting iterations."""
        if not self.estimators_:
            return
        
        # Average feature importances across all Random Forests
        importances = np.zeros(self.n_features_in_)
        for rf in self.estimators_:
            importances += rf.feature_importances_
        
        self.feature_importances_ = importances / len(self.estimators_)
    
    def predict(self, X):
        """
        Predict using the fitted GBRF model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Start with initial prediction
        predictions = np.full(len(X), self.initial_prediction_)
        
        # Add contributions from each Random Forest iteration
        for rf in self.estimators_:
            rf_pred = rf.predict(X)
            predictions = self._update_predictions(predictions, rf_pred)
        
        return predictions
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.
            
        Returns
        -------
        score : float
            R^2 score.
        """
        return r2_score(y, self.predict(X))
    
    def plot_feature_importances(self, top_n: Optional[int] = None, 
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)):
        """
        Plot feature importances.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top features to show. None = all.
        save_path : str, optional
            Path to save figure. None = display.
        figsize : tuple, default=(10, 6)
            Figure size.
        """
        check_is_fitted(self, 'feature_importances_')
        
        importances = pd.Series(
            self.feature_importances_,
            index=self.feature_names_
        ).sort_values(ascending=True)
        
        if top_n is not None:
            importances = importances.tail(top_n)
        
        plt.figure(figsize=figsize)
        importances.plot(kind='barh', color='steelblue')
        plt.title('GBRF Feature Importances (Aggregated across boosting iterations)')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6)):
        """
        Plot training and validation loss history.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure. None = display.
        figsize : tuple, default=(10, 6)
            Figure size.
        """
        if not self.train_scores_:
            raise ValueError("Model has not been trained yet.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.train_scores_, label='Train Loss', color='blue')
        
        if self.val_scores_:
            plt.plot(self.val_scores_, label='Validation Loss', color='orange')
            if hasattr(self, 'best_iteration_'):
                plt.axvline(
                    x=self.best_iteration_,
                    color='green', linestyle='--',
                    label=f'Best Iteration ({self.best_iteration_ + 1})'
                )
        
        plt.title('GBRF Training History')
        plt.xlabel('Boosting Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, filepath: str):
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        check_is_fitted(self, 'is_fitted_')
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a saved model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
            
        Returns
        -------
        model : GBRF instance
            Loaded model.
        """
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return model
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return super().get_params(deep=deep)
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        return super().set_params(**params)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_iterations={self.n_iterations}, "
            f"n_estimators_per_iteration={self.n_estimators_per_iteration}, "
            f"learning_rate={self.learning_rate})"
        )


class GRFRegressor(_BaseGBRF, RegressorMixin):
    """
    Gradient Boosted Random Forest for Regression.
    
    This is the novel hybrid algorithm where each boosting iteration
    uses a Random Forest ensemble to learn the negative gradients (residuals).
    """
    
    _is_regressor = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_base_estimator(self):
        return RandomForestRegressor()
    
    def _get_initial_prediction(self, y):
        """Initial prediction is the mean of y."""
        return np.mean(y)
    
    def _compute_pseudo_residuals(self, y, predictions):
        """Negative gradient of MSE loss: y - predictions."""
        return y - predictions
    
    def _update_predictions(self, current_pred, rf_pred):
        """Add shrunken Random Forest prediction."""
        return current_pred + self.learning_rate * rf_pred
    
    def _compute_loss(self, y_true, y_pred):
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    def predict(self, X):
        """Predict regression targets."""
        return super().predict(X)
    
    def score(self, X, y):
        """R^2 score."""
        return r2_score(y, self.predict(X))
    
    def mse(self, X, y):
        """Mean Squared Error."""
        return mean_squared_error(y, self.predict(X))
    
    def mae(self, X, y):
        """Mean Absolute Error."""
        return mean_absolute_error(y, self.predict(X))
    
    def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        cv : int, default=5
            Number of folds.
        scoring : str, default='neg_mean_squared_error'
            Scoring metric.
            
        Returns
        -------
        scores : ndarray
            Cross-validation scores.
        """
        scores = cross_val_score(self, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        return scores
    
    def grid_search(self, X, y, param_grid, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform grid search with cross-validation.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        param_grid : dict
            Parameter grid.
        cv : int, default=5
            Number of folds.
        scoring : str, default='neg_mean_squared_error'
            Scoring metric.
            
        Returns
        -------
        best_params : dict
            Best parameters found.
        """
        grid = GridSearchCV(
            self, param_grid, cv=cv, scoring=scoring,
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        grid.fit(X, y)
        
        # Update self with best parameters
        self.set_params(**grid.best_params_)
        
        return grid.best_params_, grid.best_score_


class GRFClassifier(_BaseGBRF, ClassifierMixin):
    """
    Gradient Boosted Random Forest for Classification.
    
    Uses log-loss (cross-entropy) as the objective. Each boosting iteration
    fits a Random Forest on the pseudo-residuals (negative gradients of log-loss).
    """
    
    _is_regressor = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_base_estimator(self):
        return RandomForestRegressor()
    
    def _get_initial_prediction(self, y):
        """Initial prediction is log-odds of class probabilities."""
        # For binary: log(p/(1-p)), for multiclass: one-vs-rest
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            p = np.mean(y == self.classes_[1])
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))
        else:
            # Multiclass: return log-probabilities for each class
            probs = np.array([np.mean(y == c) for c in self.classes_])
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return np.log(probs)
    
    def _compute_pseudo_residuals(self, y, predictions):
        """Negative gradient of log-loss."""
        # Convert predictions to probabilities
        if self.n_classes_ == 2:
            probs = 1 / (1 + np.exp(-predictions))
            return y - probs  # For binary with 0/1 labels
        else:
            # Multiclass: one-vs-rest residuals
            # This is simplified; full implementation would use softmax
            exp_pred = np.exp(predictions - np.max(predictions))
            probs = exp_pred / np.sum(exp_pred)
            return y - probs  # Simplified
    
    def _update_predictions(self, current_pred, rf_pred):
        """Add shrunken Random Forest prediction."""
        return current_pred + self.learning_rate * rf_pred
    
    def _compute_loss(self, y_true, y_pred):
        """Log loss (cross-entropy)."""
        if self.n_classes_ == 2:
            probs = 1 / (1 + np.exp(-y_pred))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            y_binary = (y_true == self.classes_[1]).astype(float)
            return -np.mean(y_binary * np.log(probs) + (1 - y_binary) * np.log(1 - probs))
        else:
            # Multiclass log-loss
            exp_pred = np.exp(y_pred - np.max(y_pred, axis=0))
            probs = exp_pred / np.sum(exp_pred, axis=0)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # One-hot encode y_true
            y_onehot = np.zeros((len(y_true), self.n_classes_))
            for i, c in enumerate(self.classes_):
                y_onehot[:, i] = (y_true == c).astype(float)
            return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Get raw predictions (log-odds)
        raw_pred = np.full((len(X), self.n_classes_), self.initial_prediction_)
        
        for rf in self.estimators_:
            rf_pred = rf.predict(X)
            if rf_pred.ndim == 1:
                rf_pred = rf_pred.reshape(-1, 1)
            raw_pred += self.learning_rate * rf_pred
        
        # Convert to class labels
        if self.n_classes_ == 2:
            probs = 1 / (1 + np.exp(-raw_pred[:, 0]))
            return np.where(probs > 0.5, self.classes_[1], self.classes_[0])
        else:
            return self.classes_[np.argmax(raw_pred, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        raw_pred = np.full((len(X), self.n_classes_), self.initial_prediction_)
        
        for rf in self.estimators_:
            rf_pred = rf.predict(X)
            if rf_pred.ndim == 1:
                rf_pred = rf_pred.reshape(-1, 1)
            raw_pred += self.learning_rate * rf_pred
        
        if self.n_classes_ == 2:
            probs = 1 / (1 + np.exp(-raw_pred[:, 0]))
            return np.column_stack([1 - probs, probs])
        else:
            exp_pred = np.exp(raw_pred - np.max(raw_pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    def score(self, X, y):
        """Accuracy score."""
        return accuracy_score(y, self.predict(X))
    
    def classification_report(self, X, y):
        """Detailed classification report."""
        y_pred = self.predict(X)
        return classification_report(y, y_pred, target_names=[str(c) for c in self.classes_])
    
    def confusion_matrix(self, X, y):
        """Confusion matrix."""
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)


# Backward compatibility alias
GBRF = GRFRegressor
