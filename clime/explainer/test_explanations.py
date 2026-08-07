'''
tests that every explainer can actually produce an explanation

test_explainer.py checks that explainers predict, but get_explanation() - the
actual human facing output - was broken for the logit surrogate without any test noticing
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import pytest
import clime
from clime.explainer import AVAILABLE_EXPLAINERS

# explainers that do not build a linear surrogate with feature coefficients
NO_FEATURE_IMPORTANCES = ['Kernel SHAP', 'LIME (original)']


@pytest.fixture(scope='module')
def setup():
    train_data, test_data = clime.data.AVAILABLE_DATASETS['Gaussian'](
        class_samples=[50, 50],
        gaussian_means=[[-1, -1], [1, 1]],
        gaussian_covs=[[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    clf = clime.models.random_forest(train_data)
    return clf, train_data, test_data


@pytest.mark.parametrize('name', [n for n in AVAILABLE_EXPLAINERS
                                  if n not in NO_FEATURE_IMPORTANCES])
def test_get_explanation(name, setup):
    clf, train_data, test_data = setup
    expl = AVAILABLE_EXPLAINERS[name](clf,
                                      query_point=test_data['X'][0, :],
                                      train_data=train_data,
                                      test_data=test_data,
                                      samples=500)
    explanation = np.asarray(expl.get_explanation())
    assert explanation.ndim == 1, 'one importance per feature'
    assert explanation.shape[0] == test_data['X'].shape[1]
    assert np.isfinite(explanation).all()


@pytest.mark.parametrize('name', list(AVAILABLE_EXPLAINERS))
def test_predict_proba_is_a_probability(name, setup):
    clf, train_data, test_data = setup
    expl = AVAILABLE_EXPLAINERS[name](clf,
                                      query_point=test_data['X'][0, :],
                                      train_data=train_data,
                                      test_data=test_data,
                                      samples=500)
    probs = np.asarray(expl.predict_proba(test_data['X'][:10, :]))
    assert probs.shape == (10, 2)
    assert (probs >= 0).all() and (probs <= 1).all(), 'probabilities must be in [0, 1]'
