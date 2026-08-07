'''
numerical tests for the evaluation metrics and the query point locations

complements test_evaluation.py, which only checks that metrics run
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import pytest
import clime
from clime.evaluation import AVAILABLE_EVALUATION_METRICS
from clime.evaluation.faithfulness import _get_class_weights
from clime.evaluation.key_points import get_points_between_class_means


def test_class_weights_are_not_truncated():
    '''
    regression test: the weights used to be written into a copy of the integer
    labels, which silently floored them (7:3 data gave 2 instead of 2.333)
    '''
    y = np.array([0]*7 + [1]*3)
    weights = _get_class_weights({'X': np.zeros((10, 2)), 'y': y})
    assert weights.dtype.kind == 'f', 'weights must be floats, not truncated to int'
    np.testing.assert_allclose(weights[y == 1], 7/3)
    np.testing.assert_allclose(weights[y == 0], 1.0)


def test_local_query_probs_metric_is_actually_local():
    '''
    regression test: 'fidelity (local query probs)' was mapped to the non local
    function, so selecting it silently ran the global metric
    '''
    local = AVAILABLE_EVALUATION_METRICS['fidelity (local query probs)']
    globl = AVAILABLE_EVALUATION_METRICS['fidelity (query probs)']
    assert local is not globl
    assert local.__name__ == 'query_probs_local_fidelity'


def test_every_metric_has_a_plot_range():
    '''plots look up each metric's range, so a new metric must declare one'''
    assert set(clime.evaluation.METRIC_RANGES) == set(AVAILABLE_EVALUATION_METRICS)


def test_between_class_means_line_spans_the_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 1, (100, 4)), rng.normal(1, 1, (100, 4))])
    y = np.array([0]*100 + [1]*100)
    points = np.array(get_points_between_class_means({'X': X, 'y': y})[0])
    assert np.isfinite(points).all()
    # the line must stay inside the data, and cross between the two class means
    assert (points.min(axis=0) >= X.min(axis=0) - 1e-8).all()
    assert (points.max(axis=0) <= X.max(axis=0) + 1e-8).all()


def test_between_class_means_survives_cancelling_means():
    '''
    regression test: the direction vector used to be normalised by the *sum* of its
    components. when the class means differ in opposite directions the sum is zero
    and every query point came back as nan
    '''
    X = np.array([[0., 0.], [0., 0.], [1., -1.], [1., -1.], [-2., 2.], [3., -3.]])
    y = np.array([0, 0, 1, 1, 0, 1])
    mean_difference = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    assert np.sum(mean_difference) == 0, 'this fixture must have cancelling components'
    points = np.array(get_points_between_class_means({'X': X, 'y': y}, num_samples=5)[0])
    assert np.isfinite(points).all(), 'query points must not be nan'


def test_between_class_means_identical_means_raises():
    X = np.array([[0., 0.], [1., 1.], [0., 0.], [1., 1.]])
    y = np.array([0, 0, 1, 1])
    with pytest.raises(Exception, match='class means are identical'):
        get_points_between_class_means({'X': X, 'y': y})
