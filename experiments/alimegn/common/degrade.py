'''
black boxes whose predictions deliberately depart from the data

aLIMEgn's premise is that P(X, yhat) and P(X, y) come apart when the black box has not
learned the data well. On the datasets and models this repo uses they barely do: a random
forest on Breast Cancer agrees with the labels ~95% of the time, so any y-versus-yhat
effect is invisible. To measure one, the gap has to be forced.

Three mechanisms, each registered as an ordinary black box so the pipeline treats it like
any other:

  label noise   flip a fraction of the TRAINING labels. f learns a different function from
                the one the labels describe - aLIMEgn's "weak labels" case
  underfitting  a model with too little capacity or too little training to fit the data
  imbalance     (registered as a `dataset rebalancing` entry) under-represent one class in
                training, so f's decision boundary is displaced

The test set is never touched: it keeps its true labels, which is what makes the gap
measurable.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from clime.data.processing.balance import unbalance_undersample

# (name suffix, rate) for the label noise ladder
NOISE_RATES = (0.05, 0.1, 0.2, 0.3, 0.4)
# black boxes the noise ladder is applied to: one exactly-linear, one smooth, one
# piecewise constant, so the result cannot be a property of one model family
NOISE_BASES = ('Logistic', 'MLP', 'Random Forest')


def _flip_labels(y, rate, n_classes):
    '''deterministic label noise: same seed, same flips'''
    y = np.asarray(y).astype(np.int64).copy()
    rng = np.random.default_rng([int(clime.RANDOM_SEED), int(round(rate*1000)), len(y)])
    flip = rng.random(len(y)) < rate
    # rolling to the next class label is a flip for binary data and still well defined
    # for more than two classes
    y[flip] = (y[flip] + 1) % max(n_classes, 2)
    return y


def label_noise(base_key, rate):
    '''a registered black box, trained on labels of which `rate` have been corrupted'''
    def make(data, **kwargs):
        noisy = dict(data)   # never mutate the caller's dict (FINDINGS.md B9)
        n_classes = len(np.unique(np.asarray(data['y'])))
        noisy['y'] = _flip_labels(data['y'], rate, n_classes)
        return clime.models.AVAILABLE_MODELS[base_key](data=noisy, **kwargs)
    make.__doc__ = f'{base_key} trained with {rate:.0%} of its labels flipped'
    return make


def underfit(base_key, **model_kwargs):
    '''a registered black box given too little capacity to fit the data'''
    def make(data, **kwargs):
        return clime.models.AVAILABLE_MODELS[base_key](data=data,
                                                       **{**model_kwargs, **kwargs})
    make.__doc__ = f'{base_key} with {model_kwargs}'
    return make


def class_undersample(proportions):
    '''a registered `dataset rebalancing` entry: keep only some of each class'''
    def rebalance(data, *args):
        copy = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in data.items()}   # unbalance_undersample assigns in place
        return unbalance_undersample(copy, list(proportions))
    rebalance.__doc__ = f'undersample classes to proportions {proportions}'
    return rebalance


MODELS = {}
for _base in NOISE_BASES:
    for _rate in NOISE_RATES:
        MODELS[f'{_base} (label noise {_rate:g})'] = label_noise(_base, _rate)

MODELS.update({
    # too few iterations to converge, and a two unit hidden layer
    'MLP (underfit)': underfit('MLP', max_iter=3, hidden_layer_sizes=(2,)),
    # a single split: the crudest non-trivial decision boundary
    'Decision Tree (stump)': underfit('Decision Tree', max_depth=1),
    # regularised so hard the coefficients barely leave zero
    'Logistic (over-regularised)': underfit('Logistic', C=1e-3),
    # one neighbour: fits the training data exactly and generalises badly
    'k Nearest Neighbours (k=1)': underfit('k Nearest Neighbours', n_neighbors=1),
})

DATA_BALANCING = {
    'undersample class 0 to 20%': class_undersample([0.2, 1.0]),
    'undersample class 0 to 50%': class_undersample([0.5, 1.0]),
}
