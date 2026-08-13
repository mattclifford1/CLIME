'''
Quadratic Discriminant Analysis (Quadratic decision boundary after fitting guassian bayes rule to the data)
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import clime
from clime.data.utils import costs
from clime.models import base_model


class QDA(QuadraticDiscriminantAnalysis, base_model):
    '''
    train QDA on dataset - sub class of sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset
    '''
    def __init__(self, data, **kwargs):
        # sklearn's estimator contract requires every __init__ parameter to be stored
        # unmodified under the same name: get_params() reads the signature and does
        # getattr(self, name) for each. `data` was consumed by train() and never stored,
        # which raised AttributeError once sklearn 1.2 began calling _validate_params()
        # from fit(). FINDINGS.md B17.
        self.data = data
        super().__init__()
        self.train(data)

    # Escalating fallback regularisation. sklearn checks the rank of the *regularised*
    # covariance, so the value has to be big enough to lift it numerically - 1e-4 is not,
    # 1e-2 usually is. Climb until the fit succeeds rather than guessing one value.
    FALLBACK_REG_PARAMS = (1e-2, 1e-1, 0.5)

    def train(self, data):
        try:
            self.fit(data['X'], data['y'])
            return
        except np.linalg.LinAlgError as unregularised_failure:
            # A class covariance is rank deficient - fewer samples in that class than
            # features, which happens on wide datasets (Arrhythmia has 279 features) and
            # on the tiny fixtures the pipeline tests use. scikit-learn <1.2 fitted these
            # anyway and returned an ill defined model; it now raises. Regularise rather
            # than crash, so the configuration still produces a result.
            failure = unregularised_failure
        for reg_param in self.FALLBACK_REG_PARAMS:
            self.reg_param = reg_param
            try:
                self.fit(data['X'], data['y'])
                return
            except np.linalg.LinAlgError as regularised_failure:
                failure = regularised_failure
        # N.B. a bare `raise` here would be outside the except block, and python has no
        # active exception at that point - it raises "No active exception to reraise" and
        # hides the real cause
        raise failure
