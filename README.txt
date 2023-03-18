### Summary:
* GBRF (Gradient Boosted Random Forest)
* GBRF is a Python package that implements a custom machine learning algorithm for classification tasks. It combines the strengths of two popular algorithms, Random Forest and Gradient Boosting, to achieve better accuracy and generalization on various datasets.

GBRF stands for "Gradient Boosting Regression Forest". It combines the principles of gradient boosting and random forests to build a more robust and accurate regression model. Here are some features of the gbrf package:

1. It provides a class called GBRF that wraps the GradientBoostingRegressor class from scikit-learn and adds additional functionality such as cross-validation, grid search, and feature importance plotting.
2. It has methods to fit a model, predict, and evaluate the performance of the model.
3. It provides a method to plot the feature importances of the model.
4. It provides methods to save and load the trained model to and from a file using the joblib library.
5. It is easy to use and can be integrated into other Python programs with minimal effort.
6. It can handle both regression and classification tasks.
7. It can be used for large datasets as it supports parallel processing.
8. It can be used for feature selection as it provides a method to rank the importance of features.

* The file is availabe on https://pypi.org/project/gbrf/#files

### Steps to use:
* Open the code editor like jupyter or google colab where you can install packages. Like in Jupyter type- 
```
!pip install gbrf
```
* One can also use command prompt. Open the path where python is installed and type-
```
pip install gbrf
```
* Once installation is done open your IDE and type-

``` 
from gbrf import GBRF
```
