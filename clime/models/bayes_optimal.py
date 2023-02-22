'''
get bayes optimal classifier (for guassian data)
resource: https://xavierbourretsicotte.github.io/Optimal_Bayes_Classifier.html

means/covs are estimated using gaussian assumption if they aren't given (i.e. non gaussian known dataset)
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import scipy.stats
import numpy as np
from clime.models import base_model
from sklearn.preprocessing import normalize


class Guassian_class_conditional(base_model):
    '''
    input:
        - data: dictionary with keys 'means', 'covariances' or 'X', 'y'

    returns:
        - model: base_model (sklearn type model) trained on the data distributions
    '''
    def __init__(self, data, **kwargs):
        self.train(data)

    def train(self, data):
        if 'means' in data.keys() and 'covariances' in data.keys():
            self.means = data['means']
            self.covs = data['covariances']
            self.classes = len(self.means)
        else:
            # estimate mean and cov from the data
            self.classes = len(np.unique(data['y']))
            self.means = []
            self.covs = []
            for cl in range(self.classes):
                X_c = data['X'][data['y']==cl, :]
                self.means.append(np.mean(X_c, axis=0))
                if X_c.shape[0] > 1:
                    cov = np.cov(X_c.T)
                else:
                    cov = np.eye(X_c.shape[1])
                if not np.all(np.linalg.eigvals(cov) > 0):
                    # not positive semi definite so make identiy as a quick fix
                    cov = np.eye(X_c.shape[1])
                self.covs.append(cov)

    def predict_proba(self, X):
        '''
        get the conditional for each class P(y=i | X) and normalise to make it a probability
        '''
        scores_list = []
        for cl in range(self.classes):
            score = scipy.stats.multivariate_normal.pdf(X, mean=self.means[cl], cov=self.covs[cl])
            if score.shape is ():
                score = np.expand_dims(score, axis=0)
            scores_list.append(np.expand_dims(score, axis=1))
        cond_probs = np.concatenate(scores_list, axis=1)
        probs_norm = normalize(cond_probs, axis=1, norm='l1')
        return probs_norm

    def predict(self, X):
        probs = self.predict_proba(X)
        x = np.argmax(probs, axis=1)
        return x
