'''
Reduce results/raw/<run>/ to the CSVs in results/ that the figures and tables read.

usage: uv run python summarise.py [main sweep kernel surrogate_deltas]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys
from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from common import paths, runner, stats

KEY = ['setting', 'dataset', 'family', 'seed', 'arm', 'kind', 'index']
SCORES = ['fidelity', 'balanced', 'fidelity_0', 'fidelity_1', 'auc', 'brier']
VALUES = [f'{where}_{s}' for where in ('test', 'local') for s in SCORES]
MAIN_EXPLAINERS = ['LIME', 'CLIME', 'LIME + re-threshold']
TRAINING = ['plain', 'weighted', 'SMOTE']
# a neighbourhood sample is one-sided when under 1% of it has the other black-box class
ONE_SIDED = 0.01
MINORITY_BINS = [-1, 0, 0.01, 0.05, 0.2, 0.5]
MINORITY_LABELS = ['none', 'under 1%', '1-5%', '5-20%', '20-50%']
CONFIDENCE_BINS = [0, 0.1, 0.3, 0.45, 0.5]
CONFIDENCE_LABELS = ['0.4-0.6', '0.2-0.4 or 0.6-0.8', '0.05-0.2 or 0.8-0.95', 'beyond 0.05 / 0.95']


def save(frame, name):
    path = paths.out(paths.RESULTS, f'{name}.csv')
    frame.to_csv(path, index=False, float_format='%.5f')
    print(f'wrote results/{name}.csv ({len(frame)} rows)')


def with_all(frame, column='family'):
    '''the frame plus a copy pooled over `column`'''
    return pd.concat([frame, frame.assign(**{column: 'all'})], ignore_index=True)


def blocks(scores):
    '''per (dataset, seed) block: mean over query points, worst point and range of fidelity'''
    by = ['setting', 'dataset', 'family', 'seed', 'arm', 'kind', 'explainer']
    grouped = scores.groupby(by, observed=True)
    out = grouped[VALUES].mean()
    out['worst'] = grouped['test_fidelity'].min()
    out['range'] = grouped['test_fidelity'].max() - out['worst']
    return out.reset_index()


def replication(block, arm='plain'):
    '''does CLIME beat LIME, and by how much: the CIKM'23 claims on every setting and family'''
    rows = []
    frame = with_all(block[block.arm == arm])
    for (setting, family, kind), part in frame.groupby(['setting', 'family', 'kind']):
        index = ['dataset', 'seed'] if family != 'all' else ['dataset', 'seed', 'family']
        for metric in ('test_fidelity', 'worst', 'test_balanced', 'local_fidelity',
                       'test_auc', 'test_brier'):
            wide = part.pivot_table(index=index, columns='explainer', values=metric)
            for other in ('CLIME', 'LIME + re-threshold'):
                row = stats.paired(wide, other, 'LIME', metric.endswith('brier'))
                if row:
                    rows.append({'setting': setting, 'family': family, 'kind': kind,
                                 'metric': metric, 'explainer': other, **row})
        lime = part[part.explainer == 'LIME']
        rows.append({'setting': setting, 'family': family, 'kind': kind, 'metric': 'range',
                     'explainer': 'LIME', 'mean_a': lime['range'].mean(),
                     'collapse': (lime['range'] > 0.1).mean(), 'n': len(lime)})
    return pd.DataFrame(rows)


def datasets(block, arm='plain'):
    part = block[(block.arm == arm) & block.explainer.isin(MAIN_EXPLAINERS)]
    by = ['setting', 'dataset', 'family', 'kind', 'explainer']
    return part.groupby(by, observed=True)[[*VALUES, 'worst', 'range']].mean().reset_index()


def triad(block):
    '''operating point, ranking and calibration of every explainer under every arm'''
    part = with_all(block)
    return stats.over_datasets(part, ['setting', 'family', 'arm', 'kind', 'explainer'],
                               [*VALUES, 'worst', 'range'])


def profile(scores, queries, arm='plain'):
    '''scores along the line of query points'''
    line = scores[(scores.arm == arm) & (scores.kind == 'line')
                  & scores.explainer.isin(MAIN_EXPLAINERS)]
    out = stats.over_datasets(with_all(line), ['setting', 'family', 'index', 'explainer'],
                              ['test_fidelity', 'test_balanced', 'local_fidelity'])
    q = queries[(queries.arm == arm) & (queries.kind == 'line')]
    extra = stats.over_datasets(with_all(q), ['setting', 'family', 'index'],
                                ['f_query', 'sample_minority', 'test_weighted_share_1',
                                 'test_effective_share', 'cosine'])
    return out.merge(extra.drop(columns='datasets'), on=['setting', 'family', 'index'])


def joined(scores, queries):
    '''score rows with their neighbourhood diagnostics'''
    q = queries[[*KEY, 'label', 'f_query', 'sample_minority', 'cosine', 'same_top', 'rank',
                 'test_effective_share']]
    out = scores.merge(q.drop(columns='label'), on=KEY)
    out['minority'] = pd.cut(out.sample_minority, MINORITY_BINS, labels=MINORITY_LABELS)
    out['confidence'] = pd.cut((out.f_query - 0.5).abs(), CONFIDENCE_BINS,
                               labels=CONFIDENCE_LABELS, include_lowest=True)
    return out


def by_neighbourhood(full, column, arms=('plain',)):
    '''scores by how one-sided the neighbourhood sample is, or by the black box's confidence'''
    part = full[full.arm.isin(arms) & full.explainer.isin(MAIN_EXPLAINERS)]
    by = ['setting', 'arm', 'kind', column, 'explainer']
    grouped = part.groupby(by, observed=True)
    out = grouped[['test_fidelity', 'test_balanced', 'test_auc', 'test_brier',
                   'local_fidelity', 'cosine', 'same_top']].mean()
    out['n'] = grouped.size()
    out = out.reset_index()
    total = out.groupby(['setting', 'arm', 'kind', 'explainer'])['n'].transform('sum')
    out['share'] = out['n']/total
    return out


def arms(models, block, queries):
    '''every training arm and threshold placement: the classifier and its explanations'''
    m = with_all(models)
    m['moved'] = (m.threshold - 0.5).abs() > 1e-6
    classifier = stats.over_datasets(
        m, ['setting', 'family', 'arm', 'placement'],
        ['threshold', 'moved', 'accuracy', 'balanced_accuracy', 'recall_minority',
         'recall_majority', 'g_mean', 'positive_rate', 'roc_auc'])
    wide = with_all(block[block.explainer.isin(MAIN_EXPLAINERS)]).pivot_table(
        index=['setting', 'family', 'arm', 'dataset', 'kind'], columns='explainer',
        values=['test_fidelity', 'test_balanced', 'test_auc', 'test_brier'], observed=True)
    wide.columns = [f'{metric.removeprefix("test_")}: {explainer}' for metric, explainer in wide.columns]
    explained = wide.groupby(['setting', 'family', 'arm', 'kind']).mean().reset_index()
    q = with_all(queries).assign(one_sided=lambda d: d.sample_minority < ONE_SIDED)
    neighbourhood = stats.over_datasets(q, ['setting', 'family', 'arm', 'kind'],
                                        ['one_sided', 'sample_minority', 'cosine', 'same_top',
                                         'rank']).drop(columns='datasets')
    out = explained.merge(neighbourhood, on=['setting', 'family', 'arm', 'kind'])
    return classifier.merge(out, on=['setting', 'family', 'arm'])


def by_class(full, arms_):
    '''test points of each true class: is the minority explained as well as the majority'''
    part = full[(full.kind == 'test') & full.arm.isin(arms_)
                & full.explainer.isin(MAIN_EXPLAINERS)].copy()
    part['one_sided'] = part.sample_minority < ONE_SIDED
    return stats.over_datasets(part, ['setting', 'arm', 'label', 'explainer'],
                               ['test_fidelity', 'test_balanced', 'test_fidelity_0',
                                'test_fidelity_1', 'test_auc', 'test_brier', 'local_fidelity',
                                'one_sided', 'f_query'])


def surrogate_thresholds(block, arms_=TRAINING + ['re-thresholded']):
    '''every way of cutting LIME's surrogate, under the training arms'''
    part = with_all(block[block.arm.isin(arms_)])
    return stats.over_datasets(part, ['setting', 'family', 'arm', 'kind', 'explainer'],
                               ['test_fidelity', 'test_balanced', 'local_fidelity',
                                'local_balanced', 'worst'])


def predictors(block, models, queries, arm='plain'):
    '''what CLIME's gain along the line goes with, over (dataset, family) configurations'''
    from scipy.stats import spearmanr
    by = ['setting', 'dataset', 'family']
    line = block[(block.arm == arm) & (block.kind == 'line')]
    wide = line.pivot_table(index=by, columns='explainer', values=['test_fidelity', 'range'])
    frame = pd.DataFrame({
        'gain': wide[('test_fidelity', 'CLIME')] - wide[('test_fidelity', 'LIME')],
        'lime_fidelity': wide[('test_fidelity', 'LIME')], 'lime_range': wide[('range', 'LIME')]})
    m = models[models.arm == arm].groupby(by)[['n_features', 'ratio', 'accuracy',
                                               'balanced_accuracy', 'roc_auc',
                                               'positive_rate']].mean()
    q = queries[(queries.arm == arm) & (queries.kind == 'line')].assign(
        one_sided=lambda d: d.sample_minority < ONE_SIDED,
        in_between=lambda d: d.sample_minority.between(ONE_SIDED, 0.2))
    q = q.groupby(by)[['one_sided', 'in_between', 'sample_minority',
                       'test_effective_share']].mean()
    frame = frame.join(m).join(q).reset_index()
    rows = []
    for setting, part in frame.groupby('setting'):
        for column in part.columns.drop([*by, 'gain']):
            rho = spearmanr(part.gain, part[column], nan_policy='omit')
            rows.append({'setting': setting, 'with': column, 'rho': rho.statistic,
                         'p': rho.pvalue, 'n': len(part)})
    return frame, pd.DataFrame(rows)


def main_run(run='main'):
    scores, queries, models = (runner.load(run, t) for t in ('scores', 'queries', 'models'))
    print(f'{run}: {models.groupby(["setting", "dataset", "family", "seed"]).ngroups} jobs, '
          f'{len(queries)} query points, {len(scores)} score rows, '
          f'seeds {sorted(models.seed.unique())}')
    suffix = '' if run == 'main' else f'_{run}'
    block = blocks(scores)
    full = joined(scores, queries)
    save(replication(block), f'replication{suffix}')
    save(datasets(block), f'datasets{suffix}')
    save(triad(block[block.explainer.isin(MAIN_EXPLAINERS)]), f'triad{suffix}')
    save(profile(scores, queries), f'profile{suffix}')
    save(by_neighbourhood(full, 'minority', TRAINING + ['re-thresholded', 'CP Minimax']),
         f'one_sided{suffix}')
    save(by_neighbourhood(full, 'confidence'), f'confidence{suffix}')
    save(arms(models, block, queries), f'arms{suffix}')
    save(by_class(full, TRAINING + ['re-thresholded', 'CP Minimax', 'Slacks Deltas']),
         f'by_class{suffix}')
    save(surrogate_thresholds(block), f'surrogate_thresholds{suffix}')
    configurations, correlations = predictors(block, models, queries)
    save(configurations, f'configurations{suffix}')
    save(correlations, f'predictors{suffix}')
    counts = models.groupby(['setting', 'dataset'], observed=True).agg(
        seeds=('seed', 'nunique'), ratio=('ratio', 'mean'), n_features=('n_features', 'first'),
        n_train=('n_train', 'mean'), n_test=('n_test', 'mean')).reset_index()
    save(counts, f'data{suffix}')
    failures = runner.load(run, 'failures')
    print(f'{len(failures)} failed query points')
    return block


def kernel():
    '''the locality kernel width: how local the test score is, and what CLIME gains'''
    scores, queries = runner.load('kernel', 'scores'), runner.load('kernel', 'queries')
    by = ['setting', 'family', 'kernel_width', 'kind', 'explainer']
    out = stats.over_datasets(scores[scores.explainer.isin(MAIN_EXPLAINERS)], by,
                              ['test_fidelity', 'test_balanced', 'test_auc', 'local_fidelity'])
    extra = stats.over_datasets(queries, ['setting', 'family', 'kernel_width', 'kind'],
                                ['test_effective_share', 'sample_minority', 'cosine'])
    save(out.merge(extra.drop(columns='datasets'),
                   on=['setting', 'family', 'kernel_width', 'kind']), 'kernel')


def surrogate_deltas():
    '''every deltas method cutting LIME's surrogate'''
    scores = runner.load('surrogate_deltas', 'scores')
    scores['moved'] = (scores.cut - 0.5).abs() > 1e-9
    out = stats.over_datasets(with_all(scores), ['setting', 'family', 'kind', 'explainer'],
                              ['test_fidelity', 'test_balanced', 'local_fidelity',
                               'local_balanced', 'cut', 'moved'])
    save(out, 'surrogate_deltas')


def matched():
    '''CLIME's gain split into its change of operating point and its change of fit'''
    scores = runner.load('matched', 'scores')
    block = blocks(scores)
    out = stats.over_datasets(with_all(block), ['setting', 'family', 'kind', 'explainer'],
                              [*VALUES, 'worst'])
    save(out, 'matched')
    rows = []
    for (setting, kind), part in block.groupby(['setting', 'kind']):
        for metric in ('test_fidelity', 'worst', 'test_balanced'):
            wide = part.pivot_table(index=['dataset', 'seed'], columns='explainer', values=metric)
            for a, b, name in (('CLIME', 'LIME', 'total'), ('LIME + matched cut', 'LIME', 'cut'),
                               ('CLIME', 'LIME + matched cut', 'fit')):
                rows.append({'setting': setting, 'kind': kind, 'metric': metric, 'part': name,
                             **stats.paired(wide, a, b)})
    save(pd.DataFrame(rows), 'matched_parts')
    save(pd.DataFrame({'seed': sorted(scores.seed.unique())}), 'matched_seeds')


def sweep():
    '''the imbalance ratio: the sweep runs plus the 200:20 setting of the main run'''
    frames = {t: [runner.load('sweep', t)] for t in ('scores', 'queries', 'models')}
    datasets_ = frames['models'][0].dataset.unique()
    for t in frames:
        main = runner.load('main', t)
        frames[t].append(main[(main.setting == 'imbalanced') & main.dataset.isin(datasets_)
                              & (main.seed < 5)])
    scores, queries, models = (pd.concat(frames[t], ignore_index=True)
                               for t in ('scores', 'queries', 'models'))
    ratio = {'ratio 4': 4, 'imbalanced': 10, 'ratio 20': 20, 'ratio 40': 40}
    for frame in (scores, queries, models):
        frame['setting'] = frame.setting.map(ratio)
    block = blocks(scores)
    save(arms(models, block, queries), 'sweep_arms')
    save(triad(block[block.explainer.isin(MAIN_EXPLAINERS)]), 'sweep_triad')


if __name__ == '__main__':
    RUNS = {'main': main_run, 'sweep': sweep, 'kernel': kernel,
            'surrogate_deltas': surrogate_deltas, 'matched': matched}
    for run in sys.argv[1:] or list(RUNS):
        if not any(paths.raw('main' if run == 'sweep' else run).glob('*.scores.parquet')) \
                or not any(paths.raw(run).glob('*.scores.parquet')):
            print(f'skipped {run}: no finished jobs')
            continue
        RUNS[run]()
