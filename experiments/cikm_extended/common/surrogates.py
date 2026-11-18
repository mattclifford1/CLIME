'''the surrogates compared at one query point, all trained on one neighbourhood sample'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from clime.data.utils import costs
from clime.explainer.BLIMEY import bLIMEy
from clime.models import threshold

LIME, CLIME = 'LIME', 'CLIME'
RETHRESHOLDED = 'LIME + re-threshold'
MATCHED = 'LIME + matched cut'
PREFIX = 'LIME + '
# deltas methods fast enough to fit at every query point
FAST_DELTAS = ('DKW Minimax', 'DKW Sum', 'Gaussian Predictive', 'Gaussian Band',
               'Saw-Yang-Mo Minimax', 'F Deltas')
# kernel-weighted resample of the neighbourhood that the deltas methods are fitted to
DELTAS_SAMPLE = 500


class _kept(bLIMEy):
    '''bLIMEy that keeps its neighbourhood sample and the weights it trained with'''
    def _train_surrogate(self, sampled_data):
        self.sampled = sampled_data
        super()._train_surrogate(sampled_data)

    def score(self, X):
        '''the surrogate's unclipped estimate of P(class 1)'''
        return self.surrogate_model.predict(X)[:, 1]


def balanced_threshold(score, label, weight):
    '''the threshold on `score` maximising weighted balanced accuracy; 0.5 if a class is absent'''
    total = np.array([weight[label == 0].sum(), weight[label == 1].sum()])
    if np.any(total <= 0):
        return 0.5
    order = np.argsort(score)
    s, y, w = score[order], label[order], weight[order]
    below0 = np.concatenate([[0.0], np.cumsum(w*(y == 0))])
    below1 = np.concatenate([[0.0], np.cumsum(w*(y == 1))])
    # cut k puts the k lowest scores in class 0
    balanced = (below0/total[0] + 1 - below1/total[1])/2
    edges = np.concatenate([[s[0] - 1e-9], (s[1:] + s[:-1])/2, [s[-1] + 1e-9]])
    distinct = np.concatenate([[True], s[1:] != s[:-1], [True]])
    balanced, edges = balanced[distinct], edges[distinct]
    best = np.flatnonzero(balanced >= balanced.max() - 1e-12)
    return float(edges[best[np.argmin(np.abs(edges[best] - 0.5))]])


def matched_cut(score, weight, positive_share):
    '''the cut on `score` that calls the same weighted share of the sample class 1'''
    if positive_share <= 0:
        return float(score.max() + 1e-9)
    order = np.argsort(-score)
    above = np.cumsum(weight[order])/weight.sum()
    k = min(np.searchsorted(above, positive_share - 1e-12), len(score) - 1)
    return float(score[order][k])


def build(box, query_point, test_data, deltas=FAST_DELTAS, matched=False):
    '''
    {explainer: (surrogate, threshold on its score)} plus the two fitted bLIMEy objects

    LIME and CLIME are clime's own explainers; the others are LIME's surrogate with only
    its threshold moved (by weighted balanced accuracy, or by each deltas method in
    `deltas`), so they share its coefficients.
    '''
    lime = _kept(box, query_point, test_data=test_data)
    balanced = _kept(box, query_point, test_data=test_data, class_weight_sampled=True)
    sample = lime.sampled
    score = lime.score(sample['X'])
    label = np.asarray(sample['y']).astype(np.int64)
    weight = costs.weights_based_on_distance(query_point, sample['X'])

    out = {LIME: (lime, 0.5), CLIME: (balanced, 0.5),
           RETHRESHOLDED: (lime, balanced_threshold(score, label, weight))}
    if matched:
        # LIME's surrogate cut to predict class 1 as often as CLIME does: CLIME's change of
        # operating point without its change of fit
        share = (weight*(balanced.score(sample['X']) >= 0.5)).sum()/weight.sum()
        out[MATCHED] = (lime, matched_cut(score, weight, share))
    rng = clime.utils.rng_from_point(query_point, salt='surrogate threshold sample')
    idx = rng.choice(len(score), size=DELTAS_SAMPLE, p=weight/weight.sum())
    for method in deltas:
        placed = 0.5
        if min((label[idx] == 0).sum(), (label[idx] == 1).sum()) >= 2:
            placed, _ = threshold.deltas_threshold(score[idx], label[idx], method, start=0.5)
        out[PREFIX + method] = (lime, placed)
    return out, lime, balanced
