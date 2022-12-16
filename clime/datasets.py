# author: Jonny Erskine
# email: jonathan.erskine@bristol.ac.uk

import numpy as np
import random

class GaussClass():
    def __init__(self, x, y, variance = 1, covariance=None):
        self.mean = np.array([x, y])

        if covariance is None:
            self.cov = variance * np.eye(2)
        else:
            self.cov = covariance*variance

    def gen_data(self, randomseed, size):
        rng = np.random.default_rng(randomseed)
        self.data    = np.array(rng.multivariate_normal(self.mean, self.cov, size))
