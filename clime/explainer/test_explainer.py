# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import inspect
from clime import explainer

# def test_correct_args():
#     for expl in explainer.AVAILABLE_EXPLAINERS.keys():
#         args_sig = inspect.signature(explainer.AVAILABLE_EXPLAINERS[expl])
#         args_list = list(args_sig.parameters.keys())
#         if args_list == ['args', 'kwargs'] or args_list == []:
#             assert True
#         else:
#             assert False


import numpy as np
import pytest
import clime
from clime.explainer import AVAILABLE_EXPLAINERS


def _split(n_class_0, n_class_1, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0]*n_class_0 + [1]*n_class_1)
    X = rng.normal(size=(len(y), 2)) + np.where(y[:, None] == 1, 1.0, -1.0)
    return {'X': X, 'y': y}


def test_data_class_weights_come_from_the_training_split():
    '''
    B19. 'bLIMEy (cost sensitive class)' weights each sampled point by the inverse class
    frequency of the data the black box was TRAINED on. It used to read the test split,
    the only data dict bLIMEy kept, so whenever the two splits differ in balance - a
    rebalanced training set - it applied the test set's weights instead: none at all, when
    the test set is balanced.
    '''
    train_data = _split(40, 200, seed=0)    # class 0 the minority: weights [5, 1]
    test_data = _split(100, 100, seed=1)    # balanced: would give [1, 1]
    clf = clime.models.AVAILABLE_MODELS['Logistic'](data=train_data)
    expl = AVAILABLE_EXPLAINERS['bLIMEy (cost sensitive class)'](
        clf, query_point=np.zeros(2), train_data=train_data, test_data=test_data,
        samples=500)
    # two sampled points AT the query point, so the locality kernel is exactly 1 and the
    # weights are the class weights alone; the black box calls them class 0 and class 1
    sampled = {'X': np.zeros((2, 2)), 'y': np.array([0, 1]),
               'p(y|x)': np.array([[0.9, 0.1], [0.2, 0.8]])}
    np.testing.assert_allclose(expl._get_sampled_weights(sampled), [5.0, 1.0])

    # and the test split's balance plays no part
    expl.test_data = _split(150, 30, seed=2)
    np.testing.assert_allclose(expl._get_sampled_weights(sampled), [5.0, 1.0])


def test_data_class_weights_refuse_to_guess_the_training_split():
    '''without the training data there is no global balance to read: fail, don't substitute'''
    test_data = _split(100, 100, seed=1)
    clf = clime.models.AVAILABLE_MODELS['Logistic'](data=_split(40, 200, seed=0))
    with pytest.raises(ValueError, match='training data'):
        AVAILABLE_EXPLAINERS['bLIMEy (cost sensitive class)'](
            clf, query_point=np.zeros(2), test_data=test_data, samples=100)
