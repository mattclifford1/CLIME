# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from sklearn.linear_model import LogisticRegression
from common import threshold


def model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X @ [2.0, -1.0, 0.5] + rng.normal(size=400) > 0).astype(int)
    return LogisticRegression().fit(X, y), rng.normal(size=(1000, 3))


def test_moved_model_cut_at_half_is_the_model_cut_at_the_threshold():
    clf, X = model()
    for t in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert np.array_equal(threshold.at_threshold(clf, t).predict(X),
                              (clf.predict_proba(X)[:, 1] > t).astype(int))


def test_a_move_adds_a_constant_to_the_log_odds():
    clf, X = model()
    moved = threshold.at_threshold(clf, 0.1).log_odds(X) - clf.decision_function(X)
    assert np.allclose(moved, threshold.shift_for(0.1))
