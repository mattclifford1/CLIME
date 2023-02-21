'''
Quadratic Discriminant Analysis (Quadratic decision boundary after fitting guassian bayes rule to the data)
'''
# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import sklearn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import clime
from clime.data import costs
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
        super().__init__()
        self.train(data)

    def train(self, data):
        self.fit(data['X'], data['y'])
