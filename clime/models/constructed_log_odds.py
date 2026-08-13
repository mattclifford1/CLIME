'''
black boxes whose log-odds geometry is imposed by construction, not inferred from family

The pre-registered taxonomy (experiments/logit_lime/PREREGISTRATION.md) grouped black
boxes by the analytic form of their log-odds. Statement 4 - that the quadratic group B
would outrank the smooth group C - failed, and the post-mortem was that group membership
had been argued from model family rather than measured:

  - QDA's log-odds are quadratic, but its probabilities also saturate very hard (median
    68% of locally sampled points), which is a second, uncontrolled difference.
  - a trained MLP is "smooth non-polynomial" in principle, but on real tabular data it
    frequently learns log-odds that are nearly linear.

So family membership and actual log-odds geometry came apart. The models here close that
gap by *constructing* the geometry: each is a logistic regression on a fixed feature map,
so its log-odds are exactly linear in that map and therefore exactly the intended function
of x, with saturation controlled by the usual L2 penalty rather than by the model family.

  polynomial logistic (deg 2)   log-odds exactly quadratic in x        -> group B
  RBF logistic (Nystroem)       log-odds smooth, non-polynomial        -> group C
  bagged logistic               each member exactly linear, the average of their
                                probabilities is not - a direct test of whether what
                                matters is the family or the resulting geometry
  nearest class mean            linear decision function, different inductive bias
                                from logistic regression                -> group A
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.kernel_approximation
import sklearn.ensemble
from sklearn.base import BaseEstimator, ClassifierMixin
import clime
from clime.models.log_odds_families import _sklearn_model


def _logistic(**kwargs):
    return sklearn.linear_model.LogisticRegression(
        max_iter=2000, random_state=clime.RANDOM_SEED, **kwargs)


class polynomial_logistic(_sklearn_model):
    '''logistic regression on degree 2 features: log-odds exactly quadratic in x'''
    @staticmethod
    def estimator(**kwargs):
        return sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2,
                                                              include_bias=False)),
            ('clf', _logistic(**kwargs))])


class rbf_logistic(_sklearn_model):
    '''
    logistic regression on a Nystroem RBF map: log-odds smooth but not polynomial.
    n_components is capped against the sample count at fit time by sklearn itself
    '''
    @staticmethod
    def estimator(**kwargs):
        return sklearn.pipeline.Pipeline([
            ('rbf', sklearn.kernel_approximation.Nystroem(
                gamma=0.2, n_components=100, random_state=clime.RANDOM_SEED)),
            ('clf', _logistic(**kwargs))])


class bagged_logistic(_sklearn_model):
    '''
    bagged logistic regressions. Every member has exactly linear log-odds, but bagging
    averages *probabilities*, and the average of sigmoids is not a sigmoid of any linear
    function - so the ensemble's log-odds are not linear even though every part is
    '''
    @staticmethod
    def estimator(**kwargs):
        # `estimator=`, not `base_estimator=`: sklearn renamed it in 1.2 and removed the
        # old name in 1.4
        return sklearn.ensemble.BaggingClassifier(
            estimator=_logistic(), n_estimators=25, max_samples=0.6,
            random_state=clime.RANDOM_SEED, **kwargs)


class _NearestClassMean(BaseEstimator, ClassifierMixin):
    '''
    assign to the nearer class centroid. The decision function is linear in x, so the
    log-odds are linear once passed through a softmax over negative squared distances.
    Reimplemented rather than taken from ~/Repos/projection_models, which needs
    scikit-learn >= 1.6 for validate_data and cannot be imported alongside the 1.1.3
    this repo is pinned to.
    '''
    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if sample_weight is None:
            self.means_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            self.means_ = np.stack([np.average(X[y == c], axis=0, weights=w[y == c])
                                    for c in self.classes_])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        # -||x - mu_c||^2, whose difference between classes is linear in x
        return -np.stack([np.sum((X - m)**2, axis=1) for m in self.means_], axis=1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e/e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class nearest_class_mean(_sklearn_model):
    estimator = _NearestClassMean
