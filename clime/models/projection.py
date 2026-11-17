'''
Black boxes from the thesis's shared projection_models library, behind clime's model
interface: every one also exposes its 1-D projection and the threshold on it.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from clime.models import base_model

# torch MLP: 20 epochs of at least 400 rows, the length the oversample-curriculum pilot chose
MLP_SETTINGS = dict(hidden_sizes=(64, 64), lr=1e-3, batch_size=32)
MLP_EPOCHS = 20
MLP_MIN_EPOCH_ROWS = 400


def _forest(pm, weight, seed):
    return pm.RandomForest(n_estimators=100, seed=seed, class_weight=weight, n_jobs=1)


def _logistic(pm, weight, seed):
    return pm.Linear(seed=seed, class_weight=weight, max_iter=2000)


def _svm(pm, weight, seed):
    return pm.SVM(kernel='rbf', seed=seed, class_weight=weight, probability=True)


def _boosting(pm, weight, seed):
    return pm.HistGradientBoosting(seed=seed, class_weight=weight, projection='decision')


def _knn(pm, weight, seed, rows=5):
    return pm.KNN(n_neighbors=min(5, rows))


# family -> (builder, link between projection and P(class 1), takes class weights)
FAMILIES = {
    'Random Forest': (_forest, 'margin', True),
    'Logistic': (_logistic, 'logit', True),
    'SVM': (_svm, 'logit', True),
    'Gradient Boosting': (_boosting, 'logit', True),
    'k Nearest Neighbours': (_knn, 'margin', False),
    'MLP': (None, 'logit', True),
}


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p/(1 - p))


class projection_model(base_model):
    '''
    input:
        - data: dictionary with keys 'X', 'y'
        - family: key of FAMILIES
        - balanced_training: train with balanced class weights
        - seed: defaults to clime.RANDOM_SEED

    predict is the argmax of predict_proba, and get_projection(X) > get_threshold() agrees
    with it (for the SVM the threshold is where the Platt probability crosses 1/2).
    '''
    def __init__(self, data, family='Random Forest', balanced_training=False, seed=None):
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r}; available: {list(FAMILIES)}')
        self.family = family
        self.balanced_training = balanced_training
        self.seed = int(clime.RANDOM_SEED if seed is None else seed)
        self.link = FAMILIES[family][1]
        self.train(data)

    def train(self, data):
        X = np.asarray(data['X'], dtype=np.float64)
        y = np.asarray(data['y']).astype(np.int64)
        builder, _, weighted = FAMILIES[self.family]
        if self.balanced_training and not weighted:
            raise ValueError(f'{self.family} has no class-weighted training')
        weight = 'balanced' if self.balanced_training else None
        if self.family == 'MLP':
            self.model = self._train_mlp(X, y, weight)
        else:
            import projection_models as pm
            extra = {'rows': len(y)} if self.family == 'k Nearest Neighbours' else {}
            self.model = builder(pm, weight, self.seed, **extra).fit(X, y)
        self._fit_link(X)

    def _train_mlp(self, X, y, weight):
        import torch
        from projection_models.torch import FCTrainer
        torch.set_num_threads(1)
        model = FCTrainer(input_size=X.shape[1], seed=self.seed, force_cpu=True,
                          class_weight=weight, scheduler='cosine', total_epochs=MLP_EPOCHS,
                          projection_space='logit', **MLP_SETTINGS)
        X32 = X.astype(np.float32)
        for _ in range(MLP_EPOCHS):
            model.partial_fit(X32, y, n_samples=max(len(y), MLP_MIN_EPOCH_ROWS))
        return model

    def _fit_link(self, X):
        '''logit P(class 1) = slope * projection + offset, for the logit-linked families'''
        self.slope, self.offset = 1.0, 0.0
        if self.family == 'Logistic':
            self.slope = float(np.linalg.norm(self.model.coef_))
            self.offset = float(self.model.intercept_[0])
        elif self.family == 'SVM':
            s = self.get_projection(X)[:, 0]
            p = self.predict_proba(X)[:, 1]
            keep = (p > 1e-9) & (p < 1 - 1e-9)
            self.slope, self.offset = (float(v) for v in np.polyfit(s[keep], _logit(p[keep]), 1))

    def _input(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return X.astype(np.float32) if self.family == 'MLP' else X

    def predict_proba(self, X):
        return np.asarray(self.model.predict_proba(self._input(X)), dtype=np.float64)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def get_projection(self, X):
        return np.asarray(self.model.get_projection(self._input(X)), dtype=np.float64).reshape(-1, 1)

    def get_threshold(self):
        return 0.0 if self.link == 'margin' else -self.offset/self.slope

    def probability_at(self, projection):
        '''P(class 1) of a point with this projection'''
        if self.link == 'margin':
            return float(np.clip((projection + 1)/2, 0, 1))
        return float(1/(1 + np.exp(-(self.slope*projection + self.offset))))


def _family(name):
    def build(data, **kwargs):
        return projection_model(data, family=name, **kwargs)
    build.__name__ = f'projection_model_{name}'
    return build


# trained through autograd, which does not survive the fork-based pools of the pipeline
UNREGISTERED = ('MLP',)


def available():
    '''registry entries, empty when projection_models is not installed'''
    try:
        import projection_models   # noqa: F401
    except ImportError:
        return {}
    return {f'{name} (projection)': _family(name) for name in FAMILIES
            if name not in UNREGISTERED}
