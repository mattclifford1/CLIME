# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import pytest
import clime
from clime.models import projection, threshold

pytest.importorskip('projection_models')


def _data(n0=150, n1=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-1, 1, (n0, 3)), rng.normal(1, 1, (n1, 3))])
    return {'X': X, 'y': np.r_[np.zeros(n0, int), np.ones(n1, int)]}


@pytest.mark.parametrize('family', list(projection.FAMILIES))
def test_projection_agrees_with_probability(family):
    if family == 'MLP':
        pytest.importorskip('torch')
    data, test = _data(), _data(seed=1)
    model = projection.projection_model(data, family=family, seed=0)
    assert isinstance(model, clime.models.base_model)
    prob = model.predict_proba(test['X'])[:, 1]
    score = model.get_projection(test['X'])[:, 0]
    assert np.array_equal(score > model.get_threshold(), prob > 0.5)
    assert np.array_equal(model.predict(test['X']), (prob > 0.5).astype(int))
    back = np.array([model.probability_at(s) for s in score])
    assert np.abs(back - prob).max() < 0.02


def test_moved_threshold_moves_the_class_and_nothing_else():
    data, test = _data(), _data(seed=1)
    model = projection.projection_model(data, family='Logistic', seed=0)
    prob = model.predict_proba(test['X'])[:, 1]
    moved = threshold.moved_threshold(model, 0.2)
    assert np.array_equal(moved.predict(test['X']), (prob > 0.2).astype(int))
    # the shift is monotone, so the ranking is untouched
    assert np.array_equal(np.argsort(moved.predict_proba(test['X'])[:, 1]), np.argsort(prob))
    assert np.allclose(threshold.moved_threshold(model, 0.5).predict_proba(test['X'])[:, 1],
                       np.clip(prob, threshold.EPS, 1 - threshold.EPS))


def test_rethresholding_raises_balanced_accuracy_under_imbalance():
    data, held = _data(n1=15), _data(n0=1500, n1=150, seed=1)
    model = projection.projection_model(data, family='Logistic', seed=0)

    def balanced(m):
        pred = m.predict(held['X'])
        return (np.mean(pred[held['y'] == 0] == 0) + np.mean(pred[held['y'] == 1] == 1))/2

    placed = threshold.place(model, held, 'balanced accuracy')
    assert placed.probability < 0.5
    assert balanced(placed) > balanced(model)


def test_deltas_threshold_moves_towards_the_majority():
    pytest.importorskip('deltas')
    data = _data(n1=15)
    model = projection.projection_model(data, family='Logistic', seed=0)
    placed = threshold.place(model, data, 'CP Minimax')
    assert placed.probability < 0.5
    assert placed.predict(data['X']).mean() > model.predict(data['X']).mean()


def test_unsolved_deltas_fit_keeps_the_threshold():
    pytest.importorskip('deltas')
    scores = np.r_[np.zeros(5), np.zeros(5)]
    value, moved = threshold.deltas_threshold(scores, np.r_[np.zeros(5, int), np.ones(5, int)],
                                              'Slacks Deltas', start=0.3)
    assert value == 0.3 and not moved


def test_a_point_at_the_moved_threshold_is_not_a_tie():
    data = _data()
    model = projection.projection_model(data, family='k Nearest Neighbours', seed=0)
    moved = threshold.moved_threshold(model, 0.2)
    prob = model.predict_proba(data['X'])[:, 1]
    shifted = moved.predict_proba(data['X'])[:, 1]
    assert np.any(np.isclose(prob, 0.2))
    assert np.all(shifted[np.isclose(prob, 0.2)] < 0.5)
    assert not np.any(shifted == 0.5)
    # a threshold at the floor of the probabilities: the points on it are still not ties
    floor = threshold.moved_threshold(model, 0.0).predict_proba(data['X'])[:, 1]
    assert np.all(floor[prob == 0] < 0.5) and np.all(floor[prob > 0] > 0.5)
