# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('projection_models')
pytest.importorskip('deltas')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import clime   # noqa: E402
from clime.evaluation import AVAILABLE_EVALUATION_METRICS   # noqa: E402
from clime.explainer import AVAILABLE_EXPLAINERS   # noqa: E402
from common import boxes, evaluate, scores, surrogates, tasks   # noqa: E402


@pytest.fixture(scope='module')
def task():
    return tasks.load('Diabetes Pima Indian', 'imbalanced', 0)


def test_settings_realise_their_class_counts(task):
    assert ((task.train['y'] == 0).sum(), (task.train['y'] == 1).sum()) == (200, 20)
    balanced = tasks.load('Diabetes Pima Indian', 'balanced', 0)
    assert ((balanced.train['y'] == 0).sum(), (balanced.train['y'] == 1).sum()) == (110, 110)
    assert np.allclose(task.train['X'].mean(axis=0), 0, atol=1e-9)


def test_lime_and_clime_are_the_registered_explainers(task):
    '''same surrogate and the same local fidelity as clime's own explainers and metric'''
    clime.RANDOM_SEED = 0
    built, _ = boxes.build(task, 'Random Forest', arms=('plain',), deltas=())
    box = built['plain']
    q = evaluate.query_points(task, line=20, per_class=0)[12][3]
    made, lime, balanced = surrogates.build(box, q, task.test, deltas=())
    reference = {
        surrogates.LIME: AVAILABLE_EXPLAINERS['bLIMEy (normal)'],
        surrogates.CLIME: AVAILABLE_EXPLAINERS['bLIMEy (cost sensitive sampled)']}
    metric = AVAILABLE_EVALUATION_METRICS['fidelity (local)']
    sets = evaluate._evaluation_sets(box, task, q)
    f_prob, f_class, w, X = sets['test']
    for name, make in reference.items():
        theirs = make(black_box_model=box, query_point=q, test_data=task.test)
        ours, cut = made[name]
        assert np.allclose(ours.get_explanation(), theirs.get_explanation())
        mine = scores.operating_point(ours.score(X), cut, f_class, w)['fidelity']
        assert mine == pytest.approx(metric(theirs, black_box_model=box, data=task.test,
                                            query_point=q))


def test_balanced_threshold_is_the_best_cut():
    rng = np.random.default_rng(0)
    score = rng.normal(size=300)
    label = (score + rng.normal(scale=0.7, size=300) > 0.8).astype(int)
    weight = rng.uniform(0.1, 1, size=300)

    def balanced(cut):
        pred = score >= cut
        return ((weight*(~pred)*(label == 0)).sum()/weight[label == 0].sum()
                + (weight*pred*(label == 1)).sum()/weight[label == 1].sum())/2

    best = surrogates.balanced_threshold(score, label, weight)
    assert balanced(best) >= max(balanced(c) for c in np.linspace(-3, 3, 2001)) - 1e-12
    assert surrogates.balanced_threshold(score, np.zeros(300, int), weight) == 0.5


def test_moving_the_surrogate_threshold_keeps_the_explanation(task):
    clime.RANDOM_SEED = 0
    built, _ = boxes.build(task, 'Logistic', arms=('plain',), deltas=())
    q = evaluate.query_points(task, line=20, per_class=0)[10][3]
    made, lime, _ = surrogates.build(built['plain'], q, task.test)
    for name, (surrogate, _) in made.items():
        if name != surrogates.CLIME:
            assert surrogate is lime


def test_an_unmoved_threshold_reuses_the_plain_rows(task):
    out = evaluate.run('Diabetes Pima Indian', 'balanced', 0, 'Logistic', arms=('plain',),
                       deltas=('Gaussian Neyman-Pearson', 'Gaussian Predictive'),
                       line=4, per_class=1)
    arms = {r['arm'] for r in out['scores']}
    assert arms == {'plain', 're-thresholded', 'Gaussian Neyman-Pearson', 'Gaussian Predictive'}
    assert len(out['queries']) == 4*6


def test_matched_cut_reproduces_a_positive_share():
    rng = np.random.default_rng(1)
    score, weight = rng.normal(size=500), rng.uniform(0.1, 1, size=500)
    for share in (0.0, 0.1, 0.5, 0.93):
        cut = surrogates.matched_cut(score, weight, share)
        got = (weight*(score >= cut)).sum()/weight.sum()
        assert share <= got <= share + weight.max()/weight.sum() + 1e-12
