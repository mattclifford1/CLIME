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


class _DisagreeingBlackBox:
    '''predict and predict_proba deliberately disagree, as an SVC's can under Platt scaling'''
    def predict_proba(self, X):
        p = np.full(len(X), 0.8)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


class _ConstantSurrogate:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), int(self.p >= 0.5), dtype=np.int64)

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.stack([1 - p, p], axis=1)


def test_fidelity_reads_the_black_box_probabilities_not_predict():
    '''
    B20. Fidelity compared the surrogate with black_box_model.predict, which for an SVC is
    the sign of its decision function rather than the argmax of the probabilities every
    surrogate is fitted to. A surrogate that reproduces predict_proba exactly then scored
    below 1. The black box's class is now argmax(predict_proba).
    '''
    X = np.zeros((10, 2))
    data = {'X': X}
    fid = AVAILABLE_EVALUATION_METRICS['fidelity (local)']
    q = np.zeros(2)
    assert fid(_ConstantSurrogate(0.8), black_box_model=_DisagreeingBlackBox(),
               data=data, query_point=q) == 1.0
    assert fid(_ConstantSurrogate(0.2), black_box_model=_DisagreeingBlackBox(),
               data=data, query_point=q) == 0.0


def test_black_box_class_keeps_sklearns_tie_rule():
    '''a vote fraction of exactly 0.5 is class 0, as RandomForestClassifier.predict has it'''
    from clime.evaluation.faithfulness import black_box_class

    class Tied:
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)
    assert np.all(black_box_class(Tied(), np.zeros((3, 2))) == 0)


@pytest.mark.parametrize('model', ['Logistic', 'Random Forest', 'MLP', 'Gradient Boosting',
                                   'Decision Tree', 'k Nearest Neighbours'])
def test_black_box_class_matches_predict_where_sklearn_agrees(model):
    '''the B20 change is a no-op for every estimator whose predict is the argmax'''
    from clime.evaluation.faithfulness import black_box_class
    rng = np.random.default_rng(0)
    y = np.array([0]*60 + [1]*60)
    X = rng.normal(size=(120, 2)) + np.where(y[:, None] == 1, 1.0, -1.0)
    clf = clime.models.AVAILABLE_MODELS[model](data={'X': X, 'y': y})
    Z = rng.normal(scale=2.0, size=(2000, 2))
    np.testing.assert_array_equal(black_box_class(clf, Z), np.asarray(clf.predict(Z)))
