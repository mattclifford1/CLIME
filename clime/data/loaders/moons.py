'''
Generate toy data from the moons dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn.datasets
import sklearn.utils
import clime


def get_moons(samples=200, test=False, moons_noise=0.2, **kwargs):
    '''
    sample from the half moons data distribution
    returns:
        - data: dict containing 'X', 'y'
    '''
    seed = clime.RANDOM_SEED
    if test == True:
        seed += 1
    X, y = sklearn.datasets.make_moons(n_samples=[int(samples/2)]*2,
                                       noise=moons_noise,
                                       random_state=seed,)
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    data = {'X': X, 'y':y}
    return data
