'''
logistic regresssion wrapper of sklearn that accepts probabilites as y
'''
import sklearn
import numpy as np

class logistic_regression(sklearn.linear_model.LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        y = y[:, 1]
        super().fit(X, np.round(y), **kwargs)

    def predict(self, X):
        return self.predict_proba(X)