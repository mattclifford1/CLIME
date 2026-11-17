'''black boxes: one model family under each training arm and each threshold placement'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score
from data_loaders import OVERSAMPLERS
from clime.models import projection, threshold

FAMILIES = list(projection.FAMILIES)
TRAINING_ARMS = ('plain', 'weighted', 'SMOTE')
RETHRESHOLDED = 're-thresholded'


def all_deltas():
    '''every named deltas method'''
    from deltas.methods import METHODS
    return tuple(METHODS)


class Cached:
    '''memoises predict_proba, so surrogates built on the same sample share one black-box call'''
    def __init__(self, model):
        self.model = model
        self._memo = {}

    def clear(self):
        self._memo.clear()

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        key = (X.shape, hashlib.blake2b(X.tobytes(), digest_size=16).digest())
        if key not in self._memo:
            self._memo[key] = self.model.predict_proba(X)
        return self._memo[key]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _oversampled(train, seed):
    X, y = train['X'], train['y']
    n = int((y == 0).sum() - (y == 1).sum())
    if n <= 0:
        return train
    syn = OVERSAMPLERS['SMOTE'].generate(X, y, n, seed=seed)
    return {'X': np.vstack([X, syn.X]), 'y': np.concatenate([y, syn.y])}


def _metrics(box, test):
    y = test['y']
    pred = box.predict(test['X'])
    r0 = float((pred[y == 0] == 0).mean())
    r1 = float((pred[y == 1] == 1).mean())
    return {'accuracy': float((pred == y).mean()), 'balanced_accuracy': (r0 + r1)/2,
            'recall_majority': r0, 'recall_minority': r1, 'g_mean': float(np.sqrt(r0*r1)),
            'positive_rate': float(pred.mean()),
            'roc_auc': float(roc_auc_score(y, box.predict_proba(test['X'])[:, 1]))}


def build(task, family, arms=TRAINING_ARMS, deltas=None):
    '''
    {arm: box} and one row per arm with its threshold and test metrics

    deltas: the deltas methods placed on the plain model, every one by default; an empty
    tuple leaves the training arms only.
    '''
    deltas = all_deltas() if deltas is None else deltas
    weighted = projection.FAMILIES[family][2]
    boxes, rows = {}, []
    for arm in arms:
        if arm == 'weighted' and not weighted:
            continue
        data = _oversampled(task.train, task.seed) if arm == 'SMOTE' else task.train
        model = projection.projection_model(data, family=family, seed=task.seed,
                                            balanced_training=arm == 'weighted')
        boxes[arm] = Cached(model)
        rows.append({'arm': arm, 'threshold': 0.5, 'moved': False, 'placement': 'training',
                     **_metrics(boxes[arm], task.test)})
    if not deltas:
        return boxes, rows

    plain = boxes['plain']
    start = float(plain.get_threshold())
    train_scores = plain.get_projection(task.train['X'])[:, 0]
    placed = {RETHRESHOLDED: threshold.balanced_accuracy_threshold(
        plain.get_projection(task.val['X'])[:, 0], task.val['y'], start)}
    for method in deltas:
        placed[method], _ = threshold.deltas_threshold(train_scores, task.train['y'], method, start)
    for arm, value in placed.items():
        box = threshold.moved_threshold(plain, plain.probability_at(value), value)
        boxes[arm] = box
        rows.append({'arm': arm, 'threshold': box.probability, 'moved': value != start,
                     'placement': 'validation' if arm == RETHRESHOLDED else 'deltas',
                     **_metrics(box, task.test)})
    return boxes, rows
