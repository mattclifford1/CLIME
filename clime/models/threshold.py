'''
Post-hoc threshold placement for a black box: re-thresholding on labelled data and the
deltas methods, applied as a shift of the black box's log-odds.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import contextlib
import io
import warnings
import numpy as np

EPS = 1e-6
TIE = 1e-6   # in log-odds


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p/(1 - p))


class _margin:
    '''gives any model with predict_proba the vote-margin projection P(1) - P(0)'''
    def __init__(self, model):
        self.model = model

    def get_projection(self, X):
        p = np.asarray(self.model.predict_proba(X))
        return (p[:, 1] - p[:, 0]).reshape(-1, 1)

    def get_threshold(self):
        return 0.0

    def probability_at(self, projection):
        return float(np.clip((projection + 1)/2, 0, 1))


def with_projection(model):
    return model if hasattr(model, 'probability_at') else _margin(model)


class _frozen:
    '''what the placement methods see: a 1-D score that is already the projection'''
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def fit(self, X, y=None):
        return self

    def get_projection(self, X):
        return np.asarray(X, dtype=float).reshape(-1, 1)

    def get_threshold(self):
        return self.threshold

    def get_bias(self):
        return -self.threshold

    def predict(self, X):
        return (self.get_projection(X)[:, 0] > self.threshold).astype(int)


def balanced_accuracy_threshold(scores, y, start=0.0):
    '''the threshold on a 1-D score that maximises balanced accuracy'''
    from projection_models import ThresholdCalibrator
    cal = ThresholdCalibrator(_frozen(start), strategy='balanced')
    return float(cal.fit(np.asarray(scores).reshape(-1, 1), np.asarray(y).astype(int)).threshold_)


def deltas_threshold(scores, y, method='CP Minimax', start=0.0):
    '''
    the threshold a named deltas method places on a 1-D score, and whether it moved it

    A method that cannot solve the problem keeps the starting threshold.
    '''
    from deltas.methods import METHODS
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    named = METHODS[method]
    if named.family == 'published':
        # the options deltas/experiments/run_wide.py runs the published method with
        named = named.configured(max_trials=2000, parallel=False)
    try:
        # the legacy methods print their progress
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter('ignore')
            fitted = named(_frozen(start), scores, y)
            threshold = -float(np.ravel(fitted.get_bias())[0])
    except Exception:   # noqa: BLE001 - an unsolved fit is an outcome, not an error
        return float(start), False
    if not np.isfinite(threshold):
        return float(start), False
    return threshold, bool(threshold != start)


class moved_threshold:
    '''
    a black box whose decision threshold on P(class 1) is `probability`, moved there by
    shifting its log-odds so that the new probabilities cross 1/2 at the new threshold
    '''
    def __init__(self, model, probability=0.5, projection=None):
        self.model = model
        self.probability = float(np.clip(probability, EPS, 1 - EPS))
        self.projection_threshold = projection
        # a hair past the threshold, so a point exactly on it stays class 0 and is not a tie
        self.shift = -float(_logit(self.probability)) - TIE

    def predict_proba(self, X):
        p1 = 1/(1 + np.exp(-(_logit(np.asarray(self.model.predict_proba(X))[:, 1]) + self.shift)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def get_projection(self, X):
        return with_projection(self.model).get_projection(X)

    def get_threshold(self):
        if self.projection_threshold is None:
            return with_projection(self.model).get_threshold()
        return self.projection_threshold

    def fit(self, X, y):
        self.model.fit(X, y)


def place(model, data, how='balanced accuracy'):
    '''`model` with its threshold re-placed from the labelled `data` by the rule `how`'''
    projector = with_projection(model)
    scores = projector.get_projection(data['X'])[:, 0]
    start = float(np.ravel(projector.get_threshold())[0])
    if how == 'balanced accuracy':
        threshold = balanced_accuracy_threshold(scores, data['y'], start)
    else:
        threshold, _ = deltas_threshold(scores, data['y'], how, start)
    return moved_threshold(model, projector.probability_at(threshold), threshold)


def _balancer(how):
    def build(model, data, weight=1):
        return place(model, data, how)
    return build


def available():
    '''model balancers, limited to what is installed'''
    out = {}
    try:
        import projection_models   # noqa: F401
        out['re-threshold (balanced accuracy)'] = _balancer('balanced accuracy')
    except ImportError:
        pass
    try:
        import deltas.methods   # noqa: F401
        out['deltas threshold (CP minimax)'] = _balancer('CP Minimax')
    except ImportError:
        pass
    return out
