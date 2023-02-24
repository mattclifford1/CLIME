'''
Generate guassian data
'''
# author: Jonny Erskine <jonathan.erskine@bristol.ac.uk>
# author2: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn.utils
import random
import numpy as np
import clime

def get_gaussian(samples=200,
                 gaussian_means=[[0, 0], [1, 1]],
                 gaussian_covs=[[[1, 0], [0, 1]],
                                [[1, 0], [0, 1]]],
                 test=False,
                 **kwargs):
    '''
    sample from two Gaussian dataset

    returns:
        - data: dict containing 'X', 'y'
    '''

    X = np.empty([0, 2])
    y = np.empty([0], dtype=np.int64)
    labels = [0, 1]
    for mean, cov, label in zip(gaussian_means, gaussian_covs, labels):
        # equal proportion of class samples
        class_samples = int(samples/len(labels))
        # set up current class' sampler
        gaussclass = GaussClass(mean, cov)
        # get random seed
        seed = clime.RANDOM_SEED+label
        if test == True:
            seed += 1
        # sample points
        gaussclass.gen_data(seed, class_samples)
        X = np.vstack([X, gaussclass.data])
        y = np.append(y, [label]*class_samples)
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    return {'X': X, 'y':y, 'means':gaussian_means, 'covariances':gaussian_covs}

class GaussClass():
    def __init__(self, mean, covariance):
        self.mean = np.array(mean)
        self.cov = covariance

    def gen_data(self, randomseed, size):
        rng = np.random.default_rng(randomseed)
        self.data = np.array(rng.multivariate_normal(self.mean, self.cov, size))
