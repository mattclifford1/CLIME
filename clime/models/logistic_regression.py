'''
logistic regresssion wrapper of sklearn that accepts probabilites as y
'''
import sklearn
import numpy as np

class logistic_regression(sklearn.linear_model.LogisticRegression):
    '''
    surrogate that thresholds the black box's probabilities and fits a logistic
    regression to the resulting hard labels

    N.B. away from the decision boundary the black box often predicts a single class
    over the whole local neighbourhood. sklearn cannot fit a logistic regression to one
    class, so we fall back to a constant surrogate rather than raising - those query
    points are exactly the ones a location aware explainer needs to handle
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constant_class = None

    def fit(self, X, y, **kwargs):
        y = np.round(y[:, 1])
        if len(np.unique(y)) < 2:
            # no decision boundary in the neighbourhood: predict that class everywhere
            self.constant_class = float(y[0])
            self.coef_ = np.zeros((1, X.shape[1]))
            self.intercept_ = np.zeros(1)
            return self
        self.constant_class = None
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        if self.constant_class is not None:
            p = np.full(np.asarray(X).shape[0], self.constant_class, dtype=np.float64)
            return np.stack([1-p, p], axis=1)
        return super().predict_proba(X)
