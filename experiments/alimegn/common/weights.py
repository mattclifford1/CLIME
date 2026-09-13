'''
the weighting schemes aLIMEgn compares, and the explainers that use them

A surrogate's training sample is drawn from N(q, Sigma) and labelled by the black box.
Nothing about that sample can be made to follow the data distribution by *sampling*, since
the labels always come from f - the only free choices are which points get weight and how
much. Every scheme here is one such choice, and they differ in what information they read:

  kernel only                  nothing beyond the query point            (standard LIME)
  class frequencies from yhat  the black box's predictions on the sample (CIKM'23)
  class frequencies from y     labels, globally or locally
  density ratio                unlabelled data, no labels at all

`local y` against `local yhat` is the y-versus-yhat contrast proper: the same nearby
points, the same mechanism, only the source of the class frequencies differs. That pair is
what makes aLIMEgn's claim measurable.

Reference points for the local schemes come from the black box's TRAINING data, not the
test set, so every scheme here is a method a deployer could actually run.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from sklearn.linear_model import LogisticRegression
import clime
from clime.explainer.BLIMEY import bLIMEy
from clime.data.utils import costs

# a density ratio estimated from a finite sample is unreliable in the tails; clip it so a
# single point cannot take over the fit
RATIO_CLIP = (1e-3, 1e3)
# the largest class imbalance correction we will apply. The local frequency of a class can
# be zero (far from the boundary no nearby point carries the other label), which would send
# 1/frequency to infinity
MAX_CLASS_RATIO = 1e4


def kernel_weights(query_point, X):
    return costs.weights_based_on_distance(query_point, X)


def class_weights_from_labels(labels):
    '''
    CIKM's 1/frequency class weights, normalised so the smallest is 1

    Delegates to clime's own function so that `cost sensitive sampled` and the local
    schemes here are numerically the same mechanism fed different labels.
    '''
    return costs.weight_based_on_class_imbalance(
        {'y': np.asarray(labels).astype(np.int64)})


def local_class_weights(query_point, X_ref, labels, n_classes=2):
    '''
    1/frequency class weights from labels NEAR the query point

    Frequencies are locality weighted with the same exponential kernel the surrogate
    trains under, so "near" means the same thing on both sides. A class with no local mass
    is floored at the weight of a single unit-weight point rather than producing an
    infinite correction.
    '''
    labels = np.asarray(labels).astype(np.int64)
    w = kernel_weights(query_point, X_ref)
    total = w.sum()
    mass = np.array([w[labels == c].sum() for c in range(n_classes)], dtype=np.float64)
    floor = 1.0 if total <= 0 else max(w.max(), 1e-12)
    mass = np.maximum(mass, floor)          # as if one point of the missing class existed
    weights = total/mass
    weights /= weights.min()
    return np.clip(weights, 1, MAX_CLASS_RATIO)


def density_ratio_weights(X_sampled, X_ref, random_state=None):
    '''
    estimate p_ref(x)/p_sampled(x) with a probabilistic classifier

    The surrogate is trained on points from N(q, Sigma) but judged on points from the data
    distribution. That is a covariate shift, and the textbook correction is to weight the
    training sample by the ratio of the two densities. Both densities are unknown, but the
    ratio is recoverable from a classifier that separates the two samples: with balanced
    class weights its odds estimate the ratio directly, with the sample-size factor
    already removed.

    Needs unlabelled data only - no labels, and no access to the evaluation set.
    '''
    X_sampled = np.asarray(X_sampled, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    if len(X_ref) < 2:
        return np.ones(len(X_sampled))
    X = np.vstack([X_sampled, X_ref])
    y = np.r_[np.zeros(len(X_sampled), dtype=np.int64),
              np.ones(len(X_ref), dtype=np.int64)]
    discriminator = LogisticRegression(
        class_weight='balanced', max_iter=1000,
        random_state=clime.RANDOM_SEED if random_state is None else random_state)
    discriminator.fit(X, y)
    p = discriminator.predict_proba(X_sampled)[:, 1]
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.clip(p/(1-p), *RATIO_CLIP)


class _weighting_bLIMEy(bLIMEy):
    '''
    bLIMEy with the sample weights replaced by one of the schemes above

    The locality kernel is kept in every scheme - these are corrections applied on top of
    locality, exactly as CIKM's class weights are - and `class_weights` is still set,
    because clime.evaluation.key_points records it per query point.
    '''
    scheme = None

    def __init__(self, black_box_model, query_point, train_data=None, test_data=None,
                 **kwargs):
        # reference points for the local schemes: the data the black box was trained on.
        # stored before super().__init__, which fits the surrogate before it returns
        self._reference = train_data if train_data is not None else test_data
        self._black_box = black_box_model
        super().__init__(black_box_model, query_point, test_data=test_data, **kwargs)

    def _get_sampled_weights(self, sampled_data):
        weights = kernel_weights(self.query_point, sampled_data['X'])
        self.class_weights = costs.weight_based_on_class_imbalance(sampled_data)
        return weights*self._scheme_weights(sampled_data)

    def _scheme_weights(self, sampled_data):
        raise NotImplementedError


class local_y_bLIMEy(_weighting_bLIMEy):
    '''class frequencies from the TRUE labels of nearby training points'''
    scheme = 'local y'

    def _scheme_weights(self, sampled_data):
        ref = self._reference
        if ref is None or 'y' not in ref:
            return np.ones(len(sampled_data['X']))
        class_weights = local_class_weights(self.query_point, ref['X'], ref['y'])
        # each sampled point is weighted by the class the black box gives it - that is the
        # only label a synthetic point has
        return class_weights[np.asarray(sampled_data['y']).astype(np.int64)]


class local_yhat_bLIMEy(_weighting_bLIMEy):
    '''class frequencies from the BLACK BOX's predictions on those same nearby points'''
    scheme = 'local yhat'

    def _scheme_weights(self, sampled_data):
        ref = self._reference
        if ref is None:
            return np.ones(len(sampled_data['X']))
        yhat = self._black_box.predict(ref['X'])
        class_weights = local_class_weights(self.query_point, ref['X'], yhat)
        return class_weights[np.asarray(sampled_data['y']).astype(np.int64)]


class density_ratio_bLIMEy(_weighting_bLIMEy):
    '''weights estimating p_data(x)/p_sample(x): the covariate shift correction'''
    scheme = 'density ratio'

    def _scheme_weights(self, sampled_data):
        ref = self._reference
        if ref is None:
            return np.ones(len(sampled_data['X']))
        return density_ratio_weights(sampled_data['X'], ref['X'])


AVAILABLE = {
    'bLIMEy (local y)': local_y_bLIMEy,
    'bLIMEy (local yhat)': local_yhat_bLIMEy,
    'bLIMEy (density ratio)': density_ratio_bLIMEy,
}
