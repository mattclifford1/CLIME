'''
Reduce results/raw/images_explained/ to the CSVs in results/ that its figures and tables read.

usage: uv run python summarise_images.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from common import runner, scores, stats
from summarise import ONE_SIDED, save

RUN = 'images_explained'
SETTING = ['network', 'ratio']
IMAGE = [*SETTING, 'dataset', 'seed', 'index']
PLAIN = 'plain'
LOSSES = ('brier', 'brier_balanced', 'kl', 'kl_balanced')
CLASSES = {0: 'majority', 1: 'minority'}


def values(frame):
    return [c for c in frame.select_dtypes(['number', 'bool']).columns
            if c not in (*IMAGE, 'label', 'samples')]


def blocks(frame, by):
    '''mean over the query images of each (class, seed) block'''
    return frame.groupby([*SETTING, 'dataset', 'seed', *by], observed=True)[values(frame)].mean().reset_index()


def neighbourhoods(queries):
    '''share of query images whose masks are under 1% the rarer black-box class, by true class'''
    frame = queries.assign(one_sided=queries.minority_weighted < ONE_SIDED,
                           one_sided_unweighted=queries.minority < ONE_SIDED,
                           single_class=queries.minority == 0,
                           single_class_sample=queries.sample_minority == 0,
                           label=queries.label.map(CLASSES))
    frame = pd.concat([frame, frame.assign(label='all')], ignore_index=True)
    return stats.over_datasets(frame, [*SETTING, 'arm', 'label'],
                               ['one_sided', 'one_sided_unweighted', 'single_class', 'single_class_sample',
                                'minority_weighted', 'f_class'])


def triad(full):
    '''every score of every explainer under every arm, and its skill over a constant surrogate'''
    out = stats.over_datasets(blocks(full, ['arm', 'explainer']), [*SETTING, 'arm', 'explainer'],
                              values(full))
    for suffix in ('', '_balanced'):
        out[f'skill{suffix}'] = 1 - out[f'brier{suffix}']/out[f'brier{suffix}_constant']
    return out


def paired(full, baseline='LIME'):
    '''every explainer against LIME for the network as trained, paired over (class, seed) blocks'''
    rows = []
    block = blocks(full[full.arm == full.trained], ['arm', 'explainer'])
    for key, part in block.groupby([*SETTING, 'arm']):
        for metric in values(full):
            wide = part.pivot_table(index=['dataset', 'seed'], columns='explainer', values=metric)
            for other in wide.columns.drop(baseline):
                row = stats.paired(wide, other, baseline, metric in LOSSES)
                if row:
                    rows.append({**dict(zip([*SETTING, 'arm'], key)), 'metric': metric,
                                 'explainer': other, **row})
    return pd.DataFrame(rows)


def samples(frame):
    '''cosine between the fit on n masks and the surrogate's exact fit'''
    frame = frame[frame.arm == frame.trained]
    out = stats.over_datasets(frame, [*SETTING, 'arm', 'explainer', 'samples'], ['cosine', 'same_top'])
    close = frame.assign(close=frame.cosine > 0.99)
    out['close'] = stats.over_datasets(close, [*SETTING, 'arm', 'explainer', 'samples'], ['close'])['close']
    return out


def moves(coefs, queries):
    '''how far each explanation, and each truth, moves from the plain network's at the same image'''
    plain = coefs[coefs.arm == PLAIN]
    before = queries[queries.arm == PLAIN][[*IMAGE, 'minority_weighted']].rename(
        columns={'minority_weighted': 'before'})
    key = [*IMAGE, 'explainer', 'fit']
    frame = coefs[coefs.arm != PLAIN].merge(plain[[*key, 'coef']], on=key, suffixes=('', '_plain'))
    if frame.empty:
        return pd.DataFrame()
    found = pd.DataFrame([scores.explanation_agreement(np.asarray(a), np.asarray(b))
                          for a, b in zip(frame.coef_plain, frame.coef)], index=frame.index)
    frame = pd.concat([frame.drop(columns=['coef', 'coef_plain']), found], axis=1).merge(before, on=IMAGE)
    frame['neighbourhood'] = np.where(frame.before < ONE_SIDED, 'one-sided', 'two-sided')
    frame = pd.concat([frame, frame.assign(neighbourhood='all')], ignore_index=True)
    grouped = frame.groupby([*SETTING, 'arm', 'neighbourhood', 'explainer', 'fit'], observed=True)
    out = grouped[['cosine', 'norm_ratio', 'rank']].median().add_prefix('median_')
    out['same_top'] = grouped['same_top'].mean()
    out['unchanged'] = grouped['cosine'].agg(lambda c: (c > 0.999).mean())
    out['n'] = grouped.size()
    return out.reset_index()


def checks(coefs, queries):
    '''largest coefficient gap in the two identities that must hold exactly'''
    key = [*IMAGE, 'arm']
    fits = coefs[coefs.fit == 'sample'].pivot(index=[*key, 'trained'], columns='explainer', values='coef')
    fits = fits.reset_index().merge(queries[[*key, 'sample_minority']], on=key)

    def gap(a, b):
        return max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, b))

    single = fits[fits.sample_minority == 0]
    moved = fits[fits.arm != fits.trained].merge(fits[fits.arm == fits.trained], on=[*IMAGE, 'trained'],
                                                 suffixes=('', '_trained'))
    return pd.DataFrame([{'clime_is_lime': gap(single['CLIME'], single['LIME']), 'single_class': len(single),
                          'logit_lime_moved': gap(moved['Logit-LIME'], moved['Logit-LIME_trained']),
                          'moves': len(moved), 'fits': len(fits)}])


if __name__ == '__main__':
    queries, models = runner.load(RUN, 'queries'), runner.load(RUN, 'models')
    full = runner.load(RUN, 'scores').merge(
        queries[[*IMAGE, 'arm', 'minority_weighted', 'brier_constant', 'brier_balanced_constant']],
        on=[*IMAGE, 'arm'])
    save(stats.over_datasets(models, [*SETTING, 'arm'], values(models)), 'images_models')
    save(neighbourhoods(queries), 'images_neighbourhoods')
    save(triad(full), 'images_triad')
    save(paired(full), 'images_paired')
    save(samples(runner.load(RUN, 'samples')), 'images_samples')
    coefs = runner.load(RUN, 'coefs')
    save(checks(coefs, queries), 'images_checks')
    moved = moves(coefs, queries)
    if not moved.empty:
        save(moved, 'images_moves')
    trained = models[models.arm == models.trained].groupby(['network', 'trained'])
    save(pd.DataFrame({'networks': trained.size(), 'seeds': trained.seed.nunique(),
                       'classes': trained.dataset.nunique()}).reset_index(), 'images_run')
