'''
Reduce results/raw/explanations/ to the CSVs in results/ that its figures and tables read.

usage: uv run python summarise_explanations.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import explanations, gradients, runner, stats
from summarise import KEY, ONE_SIDED, save, with_all

RUN = 'explanations'
SMOOTH = 'smooth'
OPERATING = ['fidelity', 'balanced', 'auc']
LOSSES = ['brier', 'brier_balanced', 'kl', 'kl_balanced']
WHERE = ('test', 'local')
AGREEMENT = [f'{k}_cosine' for k in ('truth_point', 'truth_mean', 'truth_mean_probability',
                                     'stability')]
VALUES = [*(f'{w}_{s}' for w in WHERE for s in [*OPERATING, *LOSSES]), *AGREEMENT,
          'truth_point_same_top', 'stability_same_top', 'coef_norm']
CONSTANTS = [f'{w}_{s}_constant' for w in WHERE for s in LOSSES]
BEATS = [f'{w}_{s}_beats_constant' for w in WHERE for s in LOSSES]
BLOCK = ['setting', 'dataset', 'family', 'seed', 'arm', 'kind', 'explainer']


def with_groups(frame):
    '''the frame plus copies pooled over every family and over the families with a gradient'''
    smooth = frame[frame.family.isin(list(gradients.SMOOTH))].assign(family=SMOOTH)
    return pd.concat([with_all(frame), smooth], ignore_index=True)


def joined(scores, queries):
    '''score rows with the constant surrogate's losses and the neighbourhood's minority share'''
    out = scores.merge(queries[[*KEY, 'sample_minority', 'f_query', *CONSTANTS]], on=KEY)
    for w in WHERE:
        for s in LOSSES:
            beats = (out[f'{w}_{s}'] < out[f'{w}_{s}_constant']).astype(float)
            out[f'{w}_{s}_beats_constant'] = beats.where(
                out[f'{w}_{s}'].notna() & out[f'{w}_{s}_constant'].notna())
    flags = [c for c in out if c.endswith('same_top')]
    out[flags] = out[flags].astype(float)
    return out


def blocks(full):
    '''mean over the query points of each (dataset, seed, family, arm, kind, explainer)'''
    return full.groupby(BLOCK, observed=True)[[*VALUES, *CONSTANTS, *BEATS]].mean().reset_index()


def triad(block):
    '''every score of every explainer under every arm, and its skill over a constant surrogate'''
    out = stats.over_datasets(with_groups(block), ['setting', 'family', 'arm', 'kind', 'explainer'],
                              [*VALUES, *CONSTANTS, *BEATS])
    for w in WHERE:
        for s in LOSSES:
            out[f'{w}_{s}_skill'] = 1 - out[f'{w}_{s}']/out[f'{w}_{s}_constant']
    return out


def paired(block, arm='plain', baseline='LIME'):
    '''every explainer against LIME on the plain black box, paired over (dataset, seed) blocks'''
    rows = []
    frame = with_groups(block[block.arm == arm])
    for (setting, family, kind), part in frame.groupby(['setting', 'family', 'kind']):
        index = ['dataset', 'seed'] if family not in ('all', SMOOTH) else ['dataset', 'seed', 'family']
        for metric in VALUES:
            wide = part.pivot_table(index=index, columns='explainer', values=metric)
            if baseline not in wide:
                continue
            lower = metric.split('_', 1)[1] in LOSSES
            for other in wide.columns.drop(baseline):
                row = stats.paired(wide, other, baseline, lower)
                if row:
                    rows.append({'setting': setting, 'family': family, 'kind': kind,
                                 'metric': metric, 'explainer': other, **row})
    return pd.DataFrame(rows)


def moves(pairs, queries, models):
    '''
    how far each explanation, and the black box's gradient, moves from the plain arm's; a
    placed threshold counts only where it left 0.5
    '''
    job = ['setting', 'dataset', 'family', 'seed']
    key = [k for k in KEY if k != 'arm']
    plain = queries[queries.arm == 'plain']
    before = plain[[*key, 'sample_minority']].rename(columns={'sample_minority': 'before'})
    frame = pairs.merge(before, on=key).merge(models[[*job, 'arm', 'threshold', 'placement']],
                                              on=[*job, 'arm'])
    frame = frame[(frame.placement == 'training') | ((frame.threshold - 0.5).abs() > 1e-6)]
    frame['neighbourhood'] = np.where(frame.before < ONE_SIDED, 'one-sided', 'two-sided')
    frame = pd.concat([frame, frame.assign(neighbourhood='all')], ignore_index=True)
    by = ['setting', 'family', 'arm', 'kind', 'neighbourhood', 'explainer']
    grouped = with_groups(frame).groupby(by, observed=True)
    out = grouped[['cosine', 'norm_ratio', 'rank']].median().add_prefix('median_')
    out['mean_cosine'] = grouped['cosine'].mean()
    out['same_top'] = grouped['same_top'].mean()
    out['unchanged'] = grouped['cosine'].agg(lambda c: (c > 0.999).mean())
    out['n'] = grouped.size()
    return out.reset_index()


def cuts(frame):
    values = [f'{w}_{s}' for w in WHERE for s in ('fidelity', 'balanced')]
    return stats.over_datasets(with_all(frame), ['setting', 'family', 'kind', 'explainer', 'cut'],
                               values)


def instruments(full, arm='plain'):
    '''rank correlation over query points of each score with the cosine to the gradient'''
    part = full[(full.arm == arm) & full.truth_point_cosine.notna()
                & full.explainer.isin(list(explanations.EXPLAINERS))]
    part = pd.concat([part, part.assign(family=SMOOTH)], ignore_index=True)
    part = pd.concat([part, part.assign(explainer='all'),
                      part[part.explainer.isin(['LIME', 'CLIME'])].assign(explainer='LIME and CLIME')],
                     ignore_index=True)
    rows = []
    for (setting, family, kind, explainer), group in part.groupby(
            ['setting', 'family', 'kind', 'explainer'], observed=True):
        for w in WHERE:
            for s in [*OPERATING, *LOSSES]:
                pair = group[[f'{w}_{s}', 'truth_point_cosine']].dropna()
                if len(pair) < 20 or pair.nunique().min() < 2:
                    continue
                rho = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
                rows.append({'setting': setting, 'family': family, 'kind': kind,
                             'explainer': explainer, 'where': w, 'score': s, 'n': len(pair),
                             # signed so that a positive value means the better score goes with the truer explanation
                             'rho': -rho if s in LOSSES else rho})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    scores, queries = runner.load(RUN, 'scores'), runner.load(RUN, 'queries')
    full = joined(scores, queries)
    block = blocks(full)
    save(triad(block), 'explanations_triad')
    save(paired(block), 'explanations_paired')
    models = runner.load(RUN, 'models')
    save(moves(runner.load(RUN, 'pairs'), queries, models), 'explanations_moves')
    save(cuts(runner.load(RUN, 'cuts')), 'explanations_cuts')
    save(instruments(full), 'explanations_instruments')
    one_sided = queries.assign(one_sided=queries.sample_minority < ONE_SIDED)
    save(stats.over_datasets(with_groups(one_sided), ['setting', 'family', 'arm', 'kind'],
                             ['one_sided', 'has_gradient']), 'explanations_neighbourhoods')
    failures = runner.load(RUN, 'failures')
    save(pd.DataFrame([{'failed': len(failures), 'explained': len(queries),
                        'jobs': len(models.drop_duplicates(['setting', 'dataset', 'family',
                                                            'seed'])),
                        'seeds': queries.seed.nunique()}]), 'explanations_run')
