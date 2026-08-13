'''
black boxes grouped by the geometry of their log-odds

The Logit-LIME study needs black boxes whose log-odds structure is known a priori, so
that a prediction can be registered before running anything:

  exactly linear in x   logistic regression, LDA (shared covariance Gaussian)
  quadratic in x        QDA, Gaussian naive Bayes (per class covariance)
  smooth, non polynomial  MLP, SVM
  piecewise constant    decision tree, random forest, k nearest neighbours

A ridge fit in logit space is exactly the linear hypothesis class, so it should recover
the first group exactly, approximate the second, and gain nothing on the last.
See experiments/logit_lime/.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn.tree
import sklearn.neighbors
import sklearn.naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import clime
from clime.data.utils import costs
from clime.models import base_model


class _sklearn_model(base_model):
    '''wrap an sklearn classifier so it is constructed from a data dict'''
    estimator = None
    default_kwargs = {}

    def __init__(self, data, balanced_training=False, **kwargs):
        self.balanced_training = balanced_training
        self.model = self.estimator(**{**self.default_kwargs, **kwargs})
        self.train(data)

    def train(self, data):
        if self.balanced_training is True:
            instance_weights = costs.get_instance_class_weights(data)
        else:
            instance_weights = None
        if instance_weights is None:
            # do not pass sample_weight at all when it is unused: a Pipeline rejects the
            # keyword outright (ValueError), even when its value is None
            self.model.fit(data['X'], data['y'])
            return
        try:
            self.model.fit(data['X'], data['y'], sample_weight=instance_weights)
        except (TypeError, ValueError):
            # not every estimator accepts sample_weight (kNN), and a Pipeline needs it
            # addressed to a named step rather than to the pipeline itself
            try:
                self.model.fit(data['X'], data['y'],
                               **{f'{self.model.steps[-1][0]}__sample_weight':
                                  instance_weights})
            except (AttributeError, TypeError, ValueError):
                self.model.fit(data['X'], data['y'])

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LDA(_sklearn_model):
    '''linear discriminant analysis - log-odds are exactly linear in x'''
    estimator = LinearDiscriminantAnalysis

    def train(self, data):
        try:
            super().train(data)
        except np.linalg.LinAlgError:
            # The pooled covariance is rank deficient - fewer samples than features,
            # which happens on wide datasets (Arrhythmia has 279) and on the tiny
            # fixtures the pipeline tests use. The default 'svd' solver raises on this;
            # 'lsqr' with automatic shrinkage is defined for it. scikit-learn <1.2 fitted
            # these anyway and returned an ill defined model. Only a fallback, so well
            # conditioned problems keep the default solver and their previous values.
            self.model = self.estimator(solver='lsqr', shrinkage='auto')
            super().train(data)


class gaussian_naive_bayes(_sklearn_model):
    '''Gaussian naive Bayes - log-odds are quadratic in x (diagonal covariance)'''
    estimator = sklearn.naive_bayes.GaussianNB


class decision_tree(_sklearn_model):
    '''a single tree - log-odds are piecewise constant'''
    estimator = sklearn.tree.DecisionTreeClassifier

    @property
    def default_kwargs(self):
        return {'random_state': clime.RANDOM_SEED}


class knn(_sklearn_model):
    '''k nearest neighbours - probabilities are a step function of the neighbour vote'''
    estimator = sklearn.neighbors.KNeighborsClassifier
    default_kwargs = {'n_neighbors': 15}
