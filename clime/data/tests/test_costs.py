'''
numerical tests for the class imbalance / distance weightings

the pipeline tests only check that runs complete, so weighting bugs that produce
well typed but wrong numbers went unnoticed - these assert the actual values
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from clime.data.utils import costs


def _data(n_class_0, n_class_1, n_features=2):
    y = np.array([0]*n_class_0 + [1]*n_class_1)
    return {'X': np.zeros((len(y), n_features)), 'y': y}


def test_class_weights_favour_minority():
    '''weight_based_on_class_imbalance returns [class 0 weight, class 1 weight]'''
    weights = costs.weight_based_on_class_imbalance(_data(8, 2))
    assert weights[1] > weights[0], 'minority class (1) must get the larger weight'
    # inversely proportional to class counts, normalised so the smallest weight is 1
    np.testing.assert_allclose(weights, [1.0, 4.0])


def test_class_weights_balanced_data():
    weights = costs.weight_based_on_class_imbalance(_data(10, 10))
    np.testing.assert_allclose(weights, [1.0, 1.0])


def test_class_weights_single_class():
    '''LIME samples far from the boundary are often all one class'''
    np.testing.assert_allclose(costs.weight_based_on_class_imbalance(_data(10, 0)), [1.0, 1.0])


def test_instance_weights_match_their_own_label():
    '''
    regression test: the weights used to be built from a label matrix with the
    columns the wrong way round, so every instance got the *other* class' weight
    and 'balanced training' models trained anti-balanced
    '''
    data = _data(8, 2)
    instance_weights = costs.get_instance_class_weights(data)
    minority = np.unique(instance_weights[data['y'] == 1])
    majority = np.unique(instance_weights[data['y'] == 0])
    assert minority.size == 1 and majority.size == 1, 'one weight per class'
    assert minority[0] > majority[0], 'minority instances must be up weighted'


def test_instance_weights_equal_sklearn_balanced():
    '''passing these as sample_weight must equal sklearn's class_weight='balanced' '''
    from sklearn.utils.class_weight import compute_sample_weight
    data = _data(8, 2)
    np.testing.assert_allclose(costs.get_instance_class_weights(data),
                               compute_sample_weight('balanced', data['y']))


def test_distance_weights_decay_with_distance():
    query_point = np.zeros(2)
    X = np.array([[0., 0.], [1., 0.], [5., 0.]])
    weights = costs.weights_based_on_distance(query_point, X)
    assert weights[0] == 1.0, 'the query point itself has weight 1'
    assert weights[0] > weights[1] > weights[2], 'weights must decay with distance'
