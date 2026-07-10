"""
hybridgbrf - Gradient Boosted Random Forest
A hybrid machine learning algorithm combining Random Forest and Gradient Boosting.
"""

__version__ = "1.0.0"
__author__ = "Abhishek Singh"
__all__ = ["GBRF", "GBRFClassifier", "GBRFRegressor"]

from .gbrf import GBRF, GRFRegressor, GRFClassifier

# Backward compatibility alias
GBRF = GRFRegressor
