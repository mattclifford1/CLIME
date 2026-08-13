'''
Multi layer perceptron
Inherits from sckit-learn classifiers
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from sklearn.neural_network import MLPClassifier
import numpy as np
import clime
from clime.models import base_model


class MLP_simple(MLPClassifier, base_model):
    '''
    train an SVM on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    *** to train with class balance weighting using the kwarg: class_weight='balanced'
    '''

    def __init__(self, data, **kwargs):
        # sklearn's estimator contract requires every __init__ parameter to be stored
        # unmodified under the same name: get_params() reads the signature and does
        # getattr(self, name) for each. `data` was consumed by train() and never stored,
        # which raised AttributeError once sklearn 1.2 began calling _validate_params()
        # from fit(). FINDINGS.md B17.
        self.data = data
        super().__init__(random_state=clime.RANDOM_SEED, 
                         learning_rate='adaptive',
                         **kwargs)
        self.train(data)

    def train(self, data):
        self.fit(data['X'], data['y'])
