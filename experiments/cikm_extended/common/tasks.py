'''datasets and the three ways each is split: at its own class ratio, balanced and imbalanced'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from data_loaders import get_dataset
from data_loaders.imbalance import SHORT_NAMES, load_problem

# the oversample-curriculum suite: real binary tabular datasets that realise 200:20
SUITE = [
    'Banknote Authentication', 'Blood Transfusion', 'Cervical Cancer',
    'Costcla Credit Scoring Kaggle 2011', 'Costcla Credit Scoring PAKDD 2009',
    'Costcla Direct Marketing', 'Default of Credit Card Clients', 'Diabetes Pima Indian',
    'EEG Eye State', 'Framingham CHD', 'German Credit', 'KC1 Software Defects',
    'MAGIC Gamma Telescope', 'Mammographic Mass', 'Ozone Level 8hr', 'Phoneme',
    'QSAR Biodegradation', 'Spambase', 'Steel Plates Fault', 'Stroke Prediction',
    'Thyroid Sick',
]
# the three UCI datasets of the CIKM'23 paper
CIKM = ['Breast Cancer Wisconsin', 'Banknote Authentication', 'Diabetes Pima Indian']
NATIVE = ['Breast Cancer Wisconsin'] + SUITE
# too few minority rows for a 110:110 development set
TOO_FEW_FOR_BALANCED = ['Blood Transfusion', 'Cervical Cancer', 'Ozone Level 8hr', 'Thyroid Sick']
BALANCED = [d for d in SUITE if d not in TOO_FEW_FOR_BALANCED]
# keeps 30 held-out minority rows at every ratio of the sweep
SWEEP = [
    'Costcla Credit Scoring Kaggle 2011', 'Costcla Credit Scoring PAKDD 2009',
    'Costcla Direct Marketing', 'Default of Credit Card Clients', 'EEG Eye State',
    'Framingham CHD', 'KC1 Software Defects', 'MAGIC Gamma Telescope', 'Ozone Level 8hr',
    'Phoneme', 'Spambase', 'Stroke Prediction', 'Thyroid Sick',
]

# setting -> (majority rows, majority:minority ratio) of the development set
CONTROLLED = {
    'balanced': (110, 1),
    'imbalanced': (200, 10),
    'ratio 4': (200, 4),
    'ratio 20': (200, 20),
    'ratio 40': (200, 40),
}
DATASETS = {'native': NATIVE, 'balanced': BALANCED, 'imbalanced': SUITE,
            'ratio 4': SWEEP, 'ratio 20': SWEEP, 'ratio 40': SWEEP}

NATIVE_MAX_ROWS = 3000
TEST_MAX_ROWS = 2000
VALIDATION = 1/3
SHORT = {**SHORT_NAMES, 'Breast Cancer Wisconsin': 'Breast Cancer'}


@dataclass
class Task:
    dataset: str
    setting: str
    seed: int
    train: dict
    val: dict
    test: dict

    @property
    def ratio(self):
        y = self.train['y']
        return float((y == 0).sum()/max((y == 1).sum(), 1))


def _standardise(train, *others):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return [(X - mean)/std for X in (train, *others)]


def _native(dataset, seed):
    '''the dataset at its own class ratio: 70% to train, a third of the rest to validation'''
    loader = get_dataset(dataset, set_seed=seed, scale=True)
    X, y = np.asarray(loader.get_X(), dtype=np.float64), np.asarray(loader.get_y()).astype(int)
    if len(y) > NATIVE_MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=NATIVE_MAX_ROWS, stratify=y,
                                      random_state=seed)
    X, X_held, y, y_held = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X_held, y_held, test_size=VALIDATION,
                                                    stratify=y_held, random_state=seed)
    return X, y, X_val, y_val, X_test, y_test


def load(dataset, setting, seed):
    if setting == 'native':
        X, y, X_val, y_val, X_test, y_test = _native(dataset, seed)
    else:
        n_majority, ratio = CONTROLLED[setting]
        p = load_problem(dataset, n_majority=n_majority, imbalance=ratio, seed=seed,
                         validation=VALIDATION)
        if (p.n_majority, p.n_minority) != (n_majority, n_majority//ratio):
            raise ValueError(f'{dataset} gives {p.n_majority}:{p.n_minority} for {setting}')
        X, y, X_val, y_val, X_test, y_test = p.X, p.y, p.X_val, p.y_val, p.X_test, p.y_test
    if len(y_test) > TEST_MAX_ROWS:
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=TEST_MAX_ROWS,
                                                stratify=y_test, random_state=seed)
    X, X_val, X_test = _standardise(np.asarray(X, dtype=np.float64), X_val, X_test)
    return Task(dataset, setting, seed, {'X': X, 'y': y}, {'X': X_val, 'y': y_val},
                {'X': X_test, 'y': y_test})
