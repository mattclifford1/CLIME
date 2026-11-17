'''paired comparisons over (dataset, seed) blocks, as in oversample-curriculum'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BLOCK = ['dataset', 'seed']
TIE = 0.005
LOWER_IS_BETTER = ('brier',)


def paired(wide, a, b, lower_is_better=False):
    '''
    column a minus column b of a frame indexed by (dataset, seed): mean difference, 95%
    interval, Wilcoxon p, and datasets where a is better, level (within 0.005) or worse
    '''
    diff = (wide[a] - wide[b]).dropna()
    if len(diff) == 0:
        return {}
    per_dataset = diff.groupby(level='dataset').mean()*(-1 if lower_is_better else 1)
    spread = diff.std(ddof=1) if len(diff) > 1 else np.nan
    return {'mean_a': wide.loc[diff.index, a].mean(), 'mean_b': wide.loc[diff.index, b].mean(),
            'diff': diff.mean(), 'ci': 1.96*spread/np.sqrt(len(diff)),
            'p': wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0,
            'n': len(diff), 'datasets': len(per_dataset),
            'wins': int((per_dataset > TIE).sum()),
            'ties': int((per_dataset.abs() <= TIE).sum()),
            'losses': int((per_dataset < -TIE).sum())}


def over_datasets(frame, by, values):
    '''mean over datasets of the dataset means, so every dataset counts once'''
    per_dataset = frame.groupby([*by, 'dataset'], observed=True)[values].mean()
    out = per_dataset.groupby(by, observed=True).mean()
    out['datasets'] = per_dataset.groupby(by, observed=True).size()
    return out.reset_index()
