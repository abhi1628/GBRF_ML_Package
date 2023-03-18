import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class GBRF:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.gbr = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_importances_ = None

    def fit(self, X, y, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.gbr = GradientBoostingRegressor(random_state=self.random_state, **kwargs)
        self.gbr.fit(X_train, y_train)
        self.y_pred = self.gbr.predict(X_test)

    def score(self):
        return r2_score(self.y_test, self.y_pred)

    def mse(self):
        return mean_squared_error(self.y_test, self.y_pred)

    def plot_feature_importances(self, **kwargs):
        feature_importances = pd.Series(self.gbr.feature_importances_, index=self.X_train.columns)
        feature_importances.nlargest(len(self.X_train.columns)).plot(kind='barh', **kwargs)

    def cross_validate(self, cv=5, **kwargs):
        scores = cross_val_score(self.gbr, self.X_train, self.y_train, cv=cv, **kwargs)
        return scores.mean()

    def grid_search(self, param_grid, cv=5, **kwargs):
        grid_search = GridSearchCV(self.gbr, param_grid, cv=cv, **kwargs)
        grid_search.fit(self.X_train, self.y_train)
        self.gbr = grid_search.best_estimator_
        self.gbr.fit(self.X_train, self.y_train)
        self.y_pred = self.gbr.predict(self.X_test)
        return grid_search.best_params_

    def save_model(self, filepath):
        joblib.dump(self.gbr, filepath)

    def load_model(self, filepath):
        self.gbr = joblib.load(filepath)
        
    def feature_importance(self):
        feature_importances = pd.Series(self.gbr.feature_importances_, index=self.X_train.columns)
        self.feature_importances_ = feature_importances.sort_values(ascending=False)
        return self.feature_importances_
    
    def predict(self, X):
        return self.gbr.predict(X)
          
   
    
