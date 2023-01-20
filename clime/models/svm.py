# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import sklearn.svm
import numpy as np
from clime.data import costs
from clime.models import base_model


class SVM(sklearn.svm.SVC, base_model):
    '''
    train an SVM on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    to train with class balance weighting using the kwarg: class_weight='balanced'
    '''
    def __init__(self, data, gamma=2, C=1, probability=True, **kwargs):
        self.data = data    # colab wont work unless we attribute data? (older python version)
        super().__init__(gamma=gamma, C=C, probability=probability, **kwargs)
        self.train(self.data)

    def train(self, data):
        self.fit(data['X'], data['y'])


class SVM_balanced_training(sklearn.svm.SVC, base_model):
    '''
    train an SVM on dataset - sub class of sklearn svm.SVC
    input:
        - data: dictionary with keys 'X', 'y'

    returns:
        - model: sklearn model trained on the dataset

    to train with class balance weighting using the kwarg: class_weight='balanced'
    '''
    def __init__(self, data, gamma=2, C=1, probability=True, class_weight='balanced'):
        self.data = data    # colab wont work unless we attribute data? (older python version)
        super().__init__(gamma=gamma, C=C, probability=probability, class_weight=class_weight)
        self.train(self.data)

    def train(self, data):
        self.fit(data['X'], data['y'])
