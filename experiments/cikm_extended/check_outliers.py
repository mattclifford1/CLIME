'''
The headline comparison with and without the splits whose standardised test set holds an
extreme value, which stretches the covariance the neighbourhood samples are drawn with.

usage: uv run python check_outliers.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from common import runner, tasks
import summarise

EXTREME = 50   # training standard deviations
SPLIT = ['setting', 'dataset', 'seed']


def largest_values(models):
    '''the largest absolute standardised test value of every split of the main run'''
    rows = []
    for setting, dataset, seed in models[SPLIT].drop_duplicates().itertuples(index=False):
        test = tasks.load(dataset, setting, int(seed)).test['X']
        rows.append({'setting': setting, 'dataset': dataset, 'seed': seed,
                     'largest': float(np.abs(test).max())})
    return pd.DataFrame(rows)


def headline(block):
    out = summarise.replication(block)
    keep = (out.family == 'all') & (out.explainer == 'CLIME') \
        & out.metric.isin(['test_fidelity', 'test_balanced'])
    return out[keep].drop(columns=['family', 'explainer', 'collapse'], errors='ignore')


if __name__ == '__main__':
    scores, models = runner.load('main', 'scores'), runner.load('main', 'models')
    splits = largest_values(models)
    splits['extreme'] = splits.largest > EXTREME
    summarise.save(splits, 'outliers')
    block = summarise.blocks(scores).merge(splits, on=SPLIT)
    by = ['setting', 'kind', 'metric']
    both = headline(block).merge(headline(block[~block.extreme]), on=by,
                                 suffixes=('', '_without'))
    counts = splits.groupby('setting').extreme.agg(extreme='sum', splits='size').reset_index()
    summarise.save(both.merge(counts, on='setting'), 'outlier_sensitivity')
