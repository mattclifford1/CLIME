'''
Why LIME misses a logistic regression's gradient: its ridge refitted at other kernel
widths, with no penalty, on ten times the samples, and to an exactly linear target.

usage: N_JOBS=4 uv run python check_logistic.py [seeds]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import sys
import warnings
from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from clime.data.utils import costs
from common import boxes, evaluate, gradients, surrogates, tasks
import summarise

SETTING = 'imbalanced'
DEFAULT = 0.75
# kernel width as a multiple of sqrt(d); infinite is no kernel
WIDTHS = {0.25: 'third', 0.5: 'two thirds', DEFAULT: 'default', 1.5: 'double', 3.0: 'fourfold',
          np.inf: 'uniform'}
MORE_SAMPLES = 100_000
CLOSE = 0.99


def cosine(a, b):
    return float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-300))


def kernel(q, X, width):
    return np.ones(len(X)) if np.isinf(width) else costs.weights_based_on_distance(q, X, width)


def fits(box, q, test, samples=None):
    '''{(fit, width): cosine with the gradient} of ridge fits on one neighbourhood sample'''
    extra = {} if samples is None else {'samples': samples}
    sample = surrogates.kept(box, q, test_data=test, **extra).sampled
    X, p = sample['X'], sample['p(y|x)'][:, 1]
    g = gradients.log_odds(box, q[None])[0]
    out = {}
    for width in ([DEFAULT] if samples else WIDTHS):
        w = kernel(q, X, width)
        out['probability', width] = cosine(Ridge(alpha=1).fit(X, p, sample_weight=w).coef_, g)
        if samples is None:
            out['linear target', width] = cosine(Ridge(alpha=1).fit(X, X @ g, sample_weight=w).coef_, g)
            out['effective share', width] = w.sum()**2/(w**2).sum()/len(w)
    if samples is None:
        w = kernel(q, X, DEFAULT)
        out['no penalty', DEFAULT] = cosine(Ridge(alpha=1e-8).fit(X, p, sample_weight=w).coef_, g)
    return out


def run(dataset, seed):
    warnings.filterwarnings('ignore')
    costs.KERNEL_WIDTH_SCALE = DEFAULT
    task = tasks.load(dataset, SETTING, seed)
    box = boxes.build(task, 'Logistic', ('plain',), ())[0]['plain']
    rows = []
    for _, index, _, q in evaluate.query_points(task, line=0):
        found = fits(box, q, task.test)
        found.update({('more samples', width): value
                      for (_, width), value in fits(box, q, task.test, MORE_SAMPLES).items()})
        rows += [{'dataset': dataset, 'seed': seed, 'index': index, 'fit': fit,
                  'width': WIDTHS[width], 'value': value} for (fit, width), value in found.items()]
    return rows


def summary(frame):
    '''each fit over datasets: mean, median and lowest dataset mean, and how many are close'''
    per_dataset = frame.groupby(['fit', 'width', 'dataset'], sort=False).value.mean()
    grouped = per_dataset.groupby(['fit', 'width'], sort=False)
    return pd.DataFrame({'mean': grouped.mean(), 'median': grouped.median(),
                         'lowest': grouped.min(),
                         'close': grouped.agg(lambda c: (c > CLOSE).sum()),
                         'datasets': grouped.size()}).reset_index()


if __name__ == '__main__':
    seeds = range(int(sys.argv[1])) if len(sys.argv) > 1 else range(5)
    todo = [(dataset, seed) for seed in seeds for dataset in tasks.DATASETS[SETTING]]
    done = Parallel(n_jobs=int(os.environ.get('N_JOBS', 4)))(delayed(run)(*job) for job in todo)
    frame = pd.DataFrame([row for rows in done for row in rows])
    summarise.save(summary(frame), 'logistic_check')
    per_dataset = frame.pivot_table(index='dataset', columns=['fit', 'width'], values='value',
                                    sort=False)
    per_dataset.columns = [f'{fit}, {width}' for fit, width in per_dataset.columns]
    summarise.save(per_dataset.reset_index(), 'logistic_check_datasets')
