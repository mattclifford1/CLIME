'''LaTeX tables from results/*.csv, and every number the write-up quotes as a macro.

Writes tables/*.tex; tables/numbers.tex defines the macros and results/numbers.txt lists them.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import re
from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from common import paths, tasks

SETTINGS = {'native': 'own ratio', 'balanced': 'balanced 110:110', 'imbalanced': 'imbalanced 200:20'}
KINDS = {'line': 'line', 'test': 'test points'}
MAIN = {'LIME': 'LIME', 'CLIME': 'CLIME', 'LIME + re-threshold': 'LIME, re-thresholded'}
METRICS = {'test_fidelity': 'Fid', 'worst': 'Worst', 'test_balanced': 'Bal',
           'local_fidelity': 'Own', 'test_auc': 'Auc', 'test_brier': 'Brier'}
TRAINING = ['plain', 'weighted', 'SMOTE', 're-thresholded']
ARM_NAMES = {'plain': 'plain (0.5)', 'weighted': 'class weighted', 'SMOTE': 'SMOTE',
             're-thresholded': 're-thresholded (validation)',
             'k-th Point Tolerance Limit (min, expected loss)': 'k-th Point Tolerance Limit'}
# letter-only names for the groups of query points by the rarer class's share of the sample
BINS = {'none': 'none', 'under 1%': 'trace', '1-5%': 'few', '5-20%': 'some', '20-50%': 'mixed'}
NUMBERS = {}


def read(name):
    path = paths.RESULTS/f'{name}.csv'
    return pd.read_csv(path) if path.exists() else None


def write(name, text):
    paths.out(paths.TABLES, f'{name}.tex').write_text(text)
    print(f'wrote tables/{name}.tex')


def macro(parts, value, fmt='{:.2f}'):
    '''register \\<parts joined in camel case> as the formatted value'''
    words = re.findall(r'[A-Za-z]+', ' '.join(str(p) for p in parts))
    name = words[0].lower() + ''.join(w[0].upper() + w[1:] for w in words[1:])
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        text = '--'
    else:
        text = fmt.format(value)
    NUMBERS[name] = text
    return text


def pct(value):
    return None if value is None or not np.isfinite(value) else 100*value


def signed(value, digits=3):
    return '--' if not np.isfinite(value) else f'{value:+.{digits}f}'


def table(columns, rows, spec=None, groups=None):
    '''
    rows: lists of cells, an empty list for a rule. groups: (label, first column, last
    column) spans drawn above the column names.
    '''
    spec = spec or 'l' + 'r'*(len(columns) - 1)
    while rows and not rows[-1]:
        rows = rows[:-1]
    body = '\n'.join('  ' + ' & '.join(r) + r' \\' if r else r'  \midrule' for r in rows)
    head = ''
    if groups:
        cells, rules, at = [], [], 1
        for label, first, last in groups:
            cells += ['']*(first - at) + [f'\\multicolumn{{{last - first + 1}}}{{c}}{{{label}}}']
            rules.append(f'\\cmidrule(lr){{{first}-{last}}}')
            at = last + 1
        head = '  ' + ' & '.join(cells) + ' \\\\\n  ' + ' '.join(rules) + '\n'
    return (f'\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n  \\toprule\n' + head + '  '
            + ' & '.join(columns) + ' \\\\\n  \\midrule\n' + body + '\n  \\bottomrule\n\\end{tabular}\n')


def datasets():
    data = read('data')
    native = data[data.setting == 'native'].set_index('dataset')
    rows = []
    for name in tasks.NATIVE:
        if name not in native.index:
            continue
        r = native.loc[name]
        rows.append([tasks.SHORT.get(name, name), f'{int(r.n_features)}', f'{r.n_train:.0f}',
                     f'{r.ratio:.1f}:1', 'yes' if name in tasks.BALANCED else '',
                     'yes' if name in tasks.SUITE else '', 'yes' if name in tasks.SWEEP else ''])
    write('datasets', table(['Dataset', 'Features', 'Training rows', 'Own ratio', '110:110',
                             '200:20', 'Sweep'], rows, 'lrrrccc'))
    macro(['datasets', 'native'], len(rows), '{:.0f}')
    macro(['seeds', 'done'], data.seeds.min(), '{:.0f}')
    for setting in SETTINGS:
        macro(['datasets', setting], (data.setting == setting).sum(), '{:.0f}')
    macro(['ratio', 'native', 'min'], native.ratio.min(), '{:.1f}')
    macro(['ratio', 'native', 'max'], native.ratio.max(), '{:.0f}')


def replication():
    frame = read('replication')
    for _, r in frame.iterrows():
        base = ['rep', r.family if r.family != 'all' else '', r.setting, r.kind]
        if r.metric == 'range':
            macro([*base, 'range'], r.mean_a)
            macro([*base, 'collapse'], pct(r.collapse), '{:.0f}')
            continue
        tag = METRICS[r.metric]
        who = 'Clime' if r.explainer == 'CLIME' else 'Reth'
        macro([*base, tag, 'lime'], r.mean_b)
        macro([*base, tag, who], r.mean_a)
        macro([*base, tag, who, 'diff'], r['diff'], '{:+.3f}')
        macro([*base, tag, who, 'ci'], r.ci, '{:.3f}')
        for count in ('wins', 'ties', 'losses', 'datasets'):
            macro([*base, tag, who, count], r[count], '{:.0f}')
    pooled = frame[(frame.family == 'all') & frame.metric.isin(['test_fidelity', 'test_balanced'])]
    rows = []
    for metric, label in (('test_fidelity', 'Local fidelity'), ('test_balanced', 'Balanced local fidelity')):
        rows.append([f'\\emph{{{label}}}'] + ['']*7)
        for setting, setting_label in SETTINGS.items():
            for kind, kind_label in KINDS.items():
                part = pooled[(pooled.metric == metric) & (pooled.setting == setting)
                              & (pooled.kind == kind)].set_index('explainer')
                if 'CLIME' not in part.index:
                    continue
                c, t = part.loc['CLIME'], part.loc['LIME + re-threshold']
                rows.append([f'{setting_label}, {kind_label}', f'{c.mean_b:.3f}', f'{c.mean_a:.3f}',
                             f'{signed(c["diff"])} $\\pm$ {c.ci:.3f}',
                             f'{c.wins:.0f}/{c.ties:.0f}/{c.losses:.0f}', f'{t.mean_a:.3f}',
                             f'{signed(t["diff"])} $\\pm$ {t.ci:.3f}',
                             f'{t.wins:.0f}/{t.ties:.0f}/{t.losses:.0f}'])
        rows.append([])
    write('replication', table(['', 'LIME', 'CLIME', 'CLIME $-$ LIME', '+/=/$-$',
                                'Re-thr.', 'Re-thr.\\ $-$ LIME', '+/=/$-$'],
                               rows, 'lrrrcrrc'))


def triad():
    frame = read('triad')
    frame = frame[(frame.family == 'all') & frame.explainer.isin(MAIN)]
    for _, r in frame.iterrows():
        base = ['tri', r.setting, r.arm, r.kind, MAIN[r.explainer].split(',')[-1]]
        for column, tag in (('test_fidelity', 'fid'), ('test_balanced', 'bal'), ('test_auc', 'auc'),
                            ('test_brier', 'brier'), ('local_fidelity', 'ownfid'),
                            ('local_balanced', 'ownbal'), ('local_auc', 'ownauc'),
                            ('local_brier', 'ownbrier')):
            macro([*base, tag], r[column], '{:.3f}' if 'brier' in tag else '{:.2f}')
    plain = frame[(frame.arm == 'plain') & (frame.kind == 'line')]
    rows = []
    for setting, label in SETTINGS.items():
        for explainer, name in MAIN.items():
            r = plain[(plain.setting == setting) & (plain.explainer == explainer)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([label if explainer == 'LIME' else '', name]
                        + [f'{r[c]:.3f}' for c in ('test_fidelity', 'test_balanced', 'test_auc',
                                                   'test_brier', 'local_fidelity',
                                                   'local_balanced', 'local_auc', 'local_brier')])
        if rows and rows[-1]:
            rows.append([])
    write('triad', table(['', '', 'Fid.', 'Bal.\\ fid.', 'AUC', 'Brier', 'Fid.', 'Bal.\\ fid.',
                          'AUC', 'Brier'], rows, 'llrrrrrrrr',
                         [('test set', 3, 6), ("surrogate's own distribution", 7, 10)]))


def arms():
    frame = read('arms')
    frame = frame[frame.family == 'all']
    for _, r in frame.iterrows():
        base = ['arm', r.setting, r.arm, r.kind]
        macro([*base, 'balacc'], r.balanced_accuracy)
        macro([*base, 'recall'], r.recall_minority)
        macro([*base, 'threshold'], r.threshold)
        macro([*base, 'moved'], pct(r.moved), '{:.0f}')
        macro([*base, 'onesided'], pct(r.one_sided), '{:.0f}')
        macro([*base, 'cosine'], r.cosine)
        macro([*base, 'sametop'], pct(r.same_top), '{:.0f}')
        for column in ('fidelity', 'balanced', 'auc', 'brier'):
            for explainer, tag in (('LIME', 'lime'), ('CLIME', 'clime'), ('LIME + re-threshold', 'reth')):
                macro([*base, column, tag], r.get(f'{column}: {explainer}', np.nan),
                      '{:.3f}' if column == 'brier' else '{:.2f}')
    for setting in ('imbalanced', 'native'):
        part = frame[(frame.setting == setting) & (frame.kind == 'test')].set_index('arm')
        if part.empty:
            continue
        order = [a for a in TRAINING if a in part.index]
        deltas_ = part[part.placement == 'deltas'].sort_values('balanced_accuracy', ascending=False)
        rows = []
        for arm in [*order, None, *deltas_.index]:
            if arm is None:
                rows.append([])
                continue
            r = part.loc[arm]
            rows.append([ARM_NAMES.get(arm, arm), f'{r.threshold:.2f}', f'{100*r.moved:.0f}',
                         f'{r.balanced_accuracy:.3f}', f'{r.recall_minority:.2f}',
                         f'{100*r.one_sided:.0f}', f'{r["fidelity: LIME"]:.3f}',
                         f'{r["balanced: LIME"]:.3f}', f'{r["balanced: CLIME"]:.3f}',
                         f'{r["balanced: LIME + re-threshold"]:.3f}'])
        write(f'arms_{setting}', table(
            ['Arm', '$t$', 'Moved \\%', 'Bal.\\ acc.', 'Min.\\ recall', 'One-sided \\%',
             'LIME fid.', 'LIME bal.', 'CLIME bal.', 'Re-thr.\\ bal.'], rows, 'lrrrrrrrrr'))


def neighbourhoods():
    frame = read('one_sided')
    plain = frame[(frame.arm == 'plain') & (frame.explainer == 'LIME')]
    for (setting, kind), part in plain.groupby(['setting', 'kind']):
        share = part.set_index('minority')['share']
        macro(['nbr', setting, kind, 'none'], pct(share.get('none', 0.0)), '{:.0f}')
        macro(['nbr', setting, kind, 'onesided'],
              pct(share.get('none', 0.0) + share.get('under 1%', 0.0)), '{:.0f}')
    for _, r in frame[frame.arm == 'plain'].iterrows():
        base = ['bin', r.setting, r.kind, BINS[r.minority], MAIN[r.explainer].split(',')[-1]]
        macro([*base, 'fid'], r.test_fidelity)
        macro([*base, 'bal'], r.test_balanced)
        macro([*base, 'cosine'], r.cosine)
        macro([*base, 'share'], pct(r.share), '{:.0f}')
    configurations = read('configurations')
    for setting, part in configurations.groupby('setting'):
        macro(['ess', setting], pct(part.test_effective_share.mean()), '{:.0f}')
    for _, r in read('predictors').iterrows():
        macro(['rho', r.setting, r['with']], r.rho, '{:+.2f}')


def by_class():
    frame = read('by_class')
    for _, r in frame.iterrows():
        base = ['cls', r.setting, r.arm, 'minority' if r.label == 1 else 'majority',
                MAIN[r.explainer].split(',')[-1]]
        macro([*base, 'fid'], r.test_fidelity)
        macro([*base, 'bal'], r.test_balanced)
        macro([*base, 'onesided'], pct(r.one_sided), '{:.0f}')


def matched():
    frame = read('matched_parts')
    for _, r in frame.iterrows():
        base = ['part', r.setting, r.kind, METRICS[r.metric], r.part]
        macro(base, r['diff'], '{:+.3f}')
        macro([*base, 'wins'], r.wins, '{:.0f}')
        macro([*base, 'losses'], r.losses, '{:.0f}')
        macro([*base, 'datasets'], r.datasets, '{:.0f}')
    macro(['matched', 'seeds'], read('matched').datasets.max()*0 + len(
        set(pd.read_csv(paths.RESULTS/'matched_seeds.csv').seed)), '{:.0f}')


def surrogate_thresholds():
    frame = read('surrogate_thresholds')
    frame = frame[(frame.family == 'all') & (frame.arm == 'plain') & (frame.kind == 'line')]
    names = list(dict.fromkeys(frame.explainer))
    order = [n for n in MAIN if n in names] + sorted(n for n in names if n not in MAIN)
    rows = []
    for explainer in order:
        row = [MAIN.get(explainer, explainer.replace('LIME + ', 'LIME, deltas: '))]
        for setting in SETTINGS:
            r = frame[(frame.setting == setting) & (frame.explainer == explainer)]
            row += ['--', '--'] if r.empty else [f'{r.test_fidelity.iloc[0]:.3f}',
                                                f'{r.test_balanced.iloc[0]:.3f}']
        rows.append(row)
    write('surrogate_thresholds', table(
        ['Surrogate', 'Fid.', 'Bal.', 'Fid.', 'Bal.', 'Fid.', 'Bal.'], rows, 'lrrrrrr',
        [(label, 2 + 2*i, 3 + 2*i) for i, label in enumerate(SETTINGS.values())]))


WIDTHS = {0.25: 'narrowest', 0.5: 'narrow', 0.75: 'default', 1.5: 'wide', 3.0: 'widest'}
RATIOS = {4: 'four', 10: 'ten', 20: 'twenty', 40: 'forty'}


def kernel():
    frame = read('kernel')
    pooled = frame.groupby(['setting', 'kernel_width', 'kind', 'explainer'])[
        ['test_fidelity', 'test_balanced', 'test_effective_share', 'sample_minority']].mean()
    for (setting, width, kind, explainer), r in pooled.iterrows():
        base = ['ker', setting, kind, WIDTHS[width], MAIN[explainer].split(',')[-1]]
        macro([*base, 'fid'], r.test_fidelity)
        macro([*base, 'bal'], r.test_balanced)
        macro([*base, 'ess'], pct(r.test_effective_share), '{:.0f}')


def sweep():
    frame = read('sweep_arms')
    frame = frame[frame.family == 'all']
    for _, r in frame.iterrows():
        base = ['swp', RATIOS[int(r.setting)], r.arm, r.kind]
        macro([*base, 'balacc'], r.balanced_accuracy)
        macro([*base, 'onesided'], pct(r.one_sided), '{:.0f}')
        for column in ('fidelity', 'balanced'):
            for explainer, tag in (('LIME', 'lime'), ('CLIME', 'clime'),
                                   ('LIME + re-threshold', 'reth')):
                macro([*base, column, tag], r.get(f'{column}: {explainer}', np.nan))


def surrogate_deltas():
    frame = read('surrogate_deltas')
    frame = frame[(frame.family == 'all') & (frame.kind == 'line')]
    wide = frame.pivot_table(index='explainer', columns='setting',
                             values=['test_fidelity', 'test_balanced', 'moved'])
    first = [n for n in MAIN if n in wide.index]
    rest = wide.drop(index=first).sort_values(('test_balanced', 'imbalanced'), ascending=False)
    rows = []
    for explainer in [*first, None, *rest.index]:
        if explainer is None:
            rows.append([])
            continue
        r = wide.loc[explainer]
        name = MAIN.get(explainer, explainer.replace('LIME + ', ''))
        rows.append([ARM_NAMES.get(name, name)] + [
            cell for setting in ('native', 'imbalanced')
            for cell in (f'{100*r[("moved", setting)]:.0f}', f'{r[("test_fidelity", setting)]:.3f}',
                         f'{r[("test_balanced", setting)]:.3f}')])
        macro(['sur', name, 'native', 'moved'], pct(r[('moved', 'native')]), '{:.0f}')
        macro(['sur', name, 'imbalanced', 'moved'], pct(r[('moved', 'imbalanced')]), '{:.0f}')
        macro(['sur', name, 'native', 'bal'], r[('test_balanced', 'native')])
        macro(['sur', name, 'imbalanced', 'bal'], r[('test_balanced', 'imbalanced')])
        macro(['sur', name, 'native', 'fid'], r[('test_fidelity', 'native')])
        macro(['sur', name, 'imbalanced', 'fid'], r[('test_fidelity', 'imbalanced')])
    write('surrogate_deltas', table(
        ['Cut of LIME\'s surrogate', 'Moved \\%', 'Fid.', 'Bal.', 'Moved \\%', 'Fid.', 'Bal.'],
        rows, 'lrrrrrr', [('own ratio', 2, 4), ('imbalanced 200:20', 5, 7)]))


def runs():
    for _, r in read('runs').iterrows():
        macro(['run', r.run, 'seeds'], r.seeds, '{:.0f}')
        macro(['run', r.run, 'jobs'], r.jobs, '{:.0f}')


def failures():
    frame = read('failures')
    macro(['failed', 'query', 'points'], frame.failed.sum(), '{:.0f}')
    macro(['explained', 'query', 'points'], frame.explained.iloc[0]/1e6, '{:.1f}')


def outliers():
    frame = read('outlier_sensitivity')
    splits = frame.drop_duplicates('setting')
    macro(['out', 'splits'], splits.extreme.sum(), '{:.0f}')
    macro(['out', 'splits', 'total'], splits.splits.sum(), '{:.0f}')
    for _, r in frame.iterrows():
        base = ['out', r.setting, r.kind, 'fid' if r.metric == 'test_fidelity' else 'bal']
        macro([*base, 'all'], r['diff'], '{:+.3f}')
        macro([*base, 'without'], r.diff_without, '{:+.3f}')


def reproduce():
    frame = read('reproduce')
    for (name, explainer), part in frame.groupby(['configuration', 'explainer']):
        macro(['pub', name.replace('±1', 'one').replace('±3', 'three'), explainer, 'mean'],
              part.fidelity.mean(), '{:.3f}')
        macro(['pub', name.replace('±1', 'one').replace('±3', 'three'), explainer, 'min'],
              part.fidelity.min())


SIX = {'LIME': 'LIME', 'CLIME': 'CLIME', 'Logit-LIME': 'Logit-LIME', 'Logit-CLIME': 'Logit-CLIME',
       'Soft logistic': 'Soft logistic', 'Soft logistic CLIME': 'Soft logistic, CLIME weights'}
GROUPS = {'all': 'all', 'smooth': 'smooth', 'Logistic': 'LR', 'SVM': 'SVM', 'MLP': 'MLP'}
EXPLANATION_SCORES = {
    'test_fidelity': 'fid', 'test_balanced': 'bal', 'test_auc': 'auc', 'test_brier': 'brier',
    'test_brier_balanced': 'brierbal', 'test_kl': 'kl', 'test_kl_balanced': 'klbal',
    'test_brier_skill': 'skill', 'test_brier_balanced_skill': 'skillbal',
    'test_kl_skill': 'klskill', 'test_kl_balanced_skill': 'klskillbal',
    'test_brier_beats_constant': 'beats', 'local_brier': 'ownbrier',
    'local_brier_balanced': 'ownbrierbal', 'local_brier_skill': 'ownskill',
    'local_brier_balanced_skill': 'ownskillbal', 'stability_cosine': 'stable',
    'stability_same_top': 'stabletop', 'coef_norm': 'norm'}
TRUTH_SCORES = {'truth_point_cosine': 'truth', 'truth_mean_cosine': 'truthmean',
                'truth_mean_probability_cosine': 'truthprob', 'truth_point_same_top': 'truthtop'}
PAIRED = {**{k: v for k, v in EXPLANATION_SCORES.items() if 'skill' not in k and 'beats' not in k},
          **TRUTH_SCORES}
MOVED_ARMS = ['re-thresholded', 'CP Minimax', 'weighted', 'SMOTE']


def _digits(tag):
    return '{:.3f}' if 'brier' in tag or 'kl' in tag else '{:.2f}'


def explanations():
    frame = read('explanations_triad')
    for _, r in frame[frame.family.isin(list(GROUPS))].iterrows():
        base = ['exp', r.setting, r.arm, r.kind, GROUPS[r.family], r.explainer]
        truths = {} if r.family == 'all' else TRUTH_SCORES
        scores = EXPLANATION_SCORES if r.family in ('all', 'smooth') and r.arm == 'plain' else \
            {k: EXPLANATION_SCORES[k] for k in ('test_fidelity', 'test_balanced', 'test_auc')} \
            if r.family == 'all' else {}
        for column, tag in {**scores, **truths}.items():
            macro([*base, tag], r[column], _digits(tag))
    plain = frame[(frame.arm == 'plain') & (frame.kind == 'test')]
    rows = []
    for setting, label in SETTINGS.items():
        for explainer, name in SIX.items():
            r = plain[(plain.setting == setting) & (plain.explainer == explainer)]
            if r.empty:
                continue
            everything, smooth = (r[r.family == f].iloc[0] for f in ('all', 'smooth'))
            rows.append([label if explainer == 'LIME' else '', name]
                        + [f'{everything[c]:.2f}' for c in (
                            'test_fidelity', 'test_balanced', 'test_auc', 'test_brier_skill',
                            'test_brier_balanced_skill', 'stability_cosine')]
                        + [f'{smooth[c]:.2f}' for c in ('truth_point_cosine', 'truth_mean_cosine')])
        if rows and rows[-1]:
            rows.append([])
    write('explanations', table(
        ['', '', 'Fid.', 'Bal.\\ fid.', 'AUC', 'Skill', 'Bal.\\ skill', 'Stability', 'at $q$',
         'local mean'], rows, 'llrrrrrrrr',
        [('agreement with the black box', 3, 7), ('cosine with gradient', 9, 10)]))


def explanations_paired():
    frame = read('explanations_paired')
    frame = frame[frame.family.isin(list(GROUPS)) & frame.metric.isin(list(PAIRED))]
    frame = frame[(frame.family == 'all') | frame.metric.isin(list(TRUTH_SCORES))
                  | (frame.metric == 'stability_cosine')]
    for _, r in frame.iterrows():
        base = ['pair', r.setting, r.kind, GROUPS[r.family], r.explainer, PAIRED[r.metric]]
        macro(base, r['diff'], '{:+.3f}')
        macro([*base, 'ci'], r.ci, '{:.3f}')
        for column in ('wins', 'losses', 'datasets'):
            macro([*base, column], r[column], '{:.0f}')


def explanations_moves():
    frame = read('explanations_moves')
    frame = frame[frame.arm.isin(MOVED_ARMS)]
    headline = (frame.setting == 'imbalanced') & (frame.kind == 'test') & (frame.neighbourhood == 'all')
    for _, r in frame[frame.family.isin(['all', 'smooth']) | headline].iterrows():
        base = ['mov', r.setting, r.kind, GROUPS.get(r.family, r.family), r.arm, r.neighbourhood,
                r.explainer]
        macro([*base, 'cos'], r.median_cosine, '{:.3f}')
        macro([*base, 'meancos'], r.mean_cosine, '{:.3f}')
        macro([*base, 'norm'], r.median_norm_ratio, '{:.2f}')
        macro([*base, 'top'], pct(r.same_top), '{:.0f}')
        macro([*base, 'changed'], pct(1 - r.same_top), '{:.0f}')
        macro([*base, 'same'], pct(r.unchanged), '{:.0f}')
    shown = frame[(frame.setting == 'imbalanced') & (frame.kind == 'test')
                  & (frame.neighbourhood == 'all')]
    rows = []
    for arm in MOVED_ARMS:
        for explainer, name in {**SIX, 'gradient': "black box's gradient"}.items():
            family = 'smooth' if explainer == 'gradient' else 'all'
            r = shown[(shown.arm == arm) & (shown.explainer == explainer) & (shown.family == family)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([ARM_NAMES.get(arm, f'deltas ({arm})') if explainer == 'LIME' else '', name,
                         f'{r.median_cosine:.3f}', f'{r.median_norm_ratio:.2f}',
                         f'{100*r.same_top:.0f}\\%', f'{100*r.unchanged:.0f}\\%'])
        rows.append([])
    write('moves', table(['Black box', 'Explanation', 'Cosine', 'Norm ratio', 'Same top feature',
                          'Unchanged'], rows, 'llrrrr'))


def explanations_spread():
    frame = read('explanations_spread')
    for _, r in frame[frame.family.isin(list(GROUPS))].iterrows():
        base = ['spr', r.setting, r.kind, GROUPS[r.family], r.explainer]
        macro([*base, 'median'], r['median'], '{:.3f}')
        macro([*base, 'lowest'], r.lowest)
        macro([*base, 'above'], r.above, '{:.0f}')
        macro([*base, 'datasets'], r.datasets, '{:.0f}')


def logistic_check():
    frame = read('logistic_check').set_index(['fit', 'width'])
    for (fit, width), r in frame.iterrows():
        base = ['lgc', fit, width]
        macro([*base, 'mean'], r['mean'], '{:.3f}')
        macro([*base, 'lowest'], r.lowest)
        macro([*base, 'close'], r.close, '{:.0f}')
        macro([*base, 'datasets'], r.datasets, '{:.0f}')
    cell = lambda fit, width: [f'{frame.loc[(fit, width), "mean"]:.3f}',
                               f'{frame.loc[(fit, width), "close"]:.0f}']
    widths = {'third': '0.25', 'two thirds': '0.5', 'default': '0.75 (LIME)', 'double': '1.5',
              'fourfold': '3', 'uniform': 'no kernel'}
    rows = [[label, f'{100*frame.loc[("effective share", width), "mean"]:.0f}\\%',
             *cell('probability', width), *cell('linear target', width)]
            for width, label in widths.items()]
    rows += [[], ['0.75, no penalty', '', *cell('no penalty', 'default'), '', '']]
    write('logistic', table(['Kernel width', 'Effective sample', 'Cosine', 'Above 0.99',
                             'Cosine', 'Above 0.99'], rows,
                            groups=[('black box\'s probability', 3, 4), ('linear target', 5, 6)]))


def logistic_samples():
    counts = {300: 'three hundred', 1000: 'thousand', 3000: 'three thousand', 10000: 'ten thousand',
              30000: 'thirty thousand', 100000: 'hundred thousand'}
    for _, r in read('logistic_samples').iterrows():
        base = ['lgs', r.explainer, counts[r.samples]]
        macro([*base, 'mean'], r['mean'], '{:.3f}')
        macro([*base, 'lowest'], r.lowest)
        macro([*base, 'close'], r.close, '{:.0f}')


IMAGE_RATIOS = {1: ('4000:4000', 'balanced'), 10: ('5000:500', 'ten'), 100: ('5000:50', 'hundred')}
IMAGE_SCORES = {'fidelity': 'fid', 'balanced': 'bal', 'auc': 'auc', 'skill': 'skill',
                'skill_balanced': 'skillbal', 'stability_cosine': 'stable',
                'converged_cosine': 'converged', 'truth_leave_one_out_cosine': 'loo',
                'truth_shapley_cosine': 'shapley', 'exact_leave_one_out_cosine': 'exactloo',
                'exact_shapley_cosine': 'exactshapley'}
IMAGE_TRUTHS = {'leave one out': 'leave-one-out effects', 'Shapley': 'Shapley values'}
IMAGE_ARMS = {'prior shift': 'prior shift', 're-thresholded': 're-thresholded (validation)',
              'CP Minimax': 'deltas (CP Minimax)', 'weighted': 'class weighted',
              'logit adjusted': 'logit-adjusted loss'}


def images():
    frame = read('images_triad')
    for _, r in frame.iterrows():
        base = ['img', r.network, IMAGE_RATIOS[r.ratio][1], r.arm, r.explainer]
        for column, tag in IMAGE_SCORES.items():
            macro([*base, tag], r[column])
    plain = frame[(frame.network == 'scratch') & (frame.arm == 'plain')]
    rows = []
    for ratio, (label, _) in IMAGE_RATIOS.items():
        for explainer, name in SIX.items():
            r = plain[(plain.ratio == ratio) & (plain.explainer == explainer)]
            if r.empty:
                continue
            rows.append([label if explainer == 'LIME' else '', name]
                        + [f'{r.iloc[0][c]:.2f}' for c in (
                            'fidelity', 'balanced', 'auc', 'skill', 'skill_balanced',
                            'stability_cosine', 'converged_cosine', 'truth_leave_one_out_cosine',
                            'truth_shapley_cosine')])
        if rows and rows[-1]:
            rows.append([])
    write('images', table(
        ['', '', 'Fid.', 'Bal.\\ fid.', 'AUC', 'Skill', 'Bal.\\ skill', 'Stability', 'exact fit',
         'leave one out', 'Shapley'], rows, 'llrrrrrrrrr',
        [('agreement with the black box', 3, 7), ('cosine with', 9, 11)]))


def images_paired():
    frame = read('images_paired')
    for _, r in frame[frame.metric.isin(list(IMAGE_SCORES))].iterrows():
        base = ['imp', r.network, IMAGE_RATIOS[r.ratio][1], r.arm, r.explainer, IMAGE_SCORES[r.metric]]
        macro(base, r['diff'], '{:+.3f}')
        macro([*base, 'ci'], r.ci, '{:.3f}')
        for column in ('wins', 'losses', 'datasets'):
            macro([*base, column], r[column], '{:.0f}')


def images_samples():
    counts = {100: 'hundred', 300: 'three hundred', 1000: 'thousand', 3000: 'three thousand',
              10000: 'ten thousand'}
    for _, r in read('images_samples').iterrows():
        base = ['ims', r.network, IMAGE_RATIOS[r.ratio][1], r.arm, r.explainer, counts[r.samples]]
        macro(base, r.cosine, '{:.3f}')
        macro([*base, 'close'], pct(r.close), '{:.0f}')


def images_moves():
    frame = read('images_moves')
    for _, r in frame.iterrows():
        base = ['imm', r.network, IMAGE_RATIOS[r.ratio][1], r.arm, r.neighbourhood, r.explainer, r.fit]
        macro([*base, 'cos'], r.median_cosine, '{:.3f}')
        macro([*base, 'norm'], r.median_norm_ratio, '{:.2f}')
        macro([*base, 'changed'], pct(1 - r.same_top), '{:.0f}')
        macro([*base, 'same'], pct(r.unchanged), '{:.0f}')
    shown = frame[(frame.network == 'scratch') & (frame.ratio == 100) & (frame.neighbourhood == 'all')
                  & frame.fit.isin(['sample', 'truth'])]
    rows = []
    for arm, label in IMAGE_ARMS.items():
        for explainer, name in {**SIX, **IMAGE_TRUTHS}.items():
            r = shown[(shown.arm == arm) & (shown.explainer == explainer)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([label if explainer == 'LIME' else '', name, f'{r.median_cosine:.3f}',
                         f'{r.median_norm_ratio:.2f}', f'{100*r.same_top:.0f}\\%',
                         f'{100*r.unchanged:.0f}\\%'])
        if rows and rows[-1]:
            rows.append([])
    write('images_moves', table(['Black box', 'Explanation', 'Cosine', 'Norm ratio', 'Same top patch',
                                 'Unchanged'], rows, 'llrrrr'))


def images_models():
    for _, r in read('images_models').iterrows():
        base = ['imo', r.network, IMAGE_RATIOS[r.ratio][1], r.arm]
        macro([*base, 'balacc'], r.balanced_accuracy)
        macro([*base, 'recall'], r.recall_minority)
        macro([*base, 'positive'], pct(r.positive_rate), '{:.1f}')
        macro([*base, 'shift'], r['shift'])
    for _, r in read('images_neighbourhoods').iterrows():
        base = ['imn', r.network, IMAGE_RATIOS[r.ratio][1], r.arm, r.label]
        macro([*base, 'onesided'], pct(r.one_sided), '{:.0f}')
    run = read('images_run').iloc[0]
    for column in run.index:
        macro(['img', 'run', column], run[column], '{:.0f}')


def explanations_instruments():
    frame = read('explanations_instruments')
    frame = frame[(frame.family == 'smooth') & frame.explainer.isin(['all', 'LIME and CLIME'])]
    for _, r in frame.iterrows():
        macro(['ins', r.setting, r.kind, r.explainer, r['where'], r.score.replace('_', ' ')],
              r.rho, '{:+.2f}')


def explanations_cuts():
    frame = read('explanations_cuts')
    frame = frame[frame.family == 'all']
    for (setting, kind, explainer), part in frame.groupby(['setting', 'kind', 'explainer']):
        base = ['cut', setting, kind, explainer]
        best = part.loc[part.test_balanced.idxmax()]
        macro([*base, 'best'], best.test_balanced)
        macro([*base, 'best', 'at'], best.cut)
        macro([*base, 'lowest'], part.test_balanced.min())
        macro([*base, 'fid', 'highest'], part.test_fidelity.max())
        macro([*base, 'fid', 'lowest'], part.test_fidelity.min())


def explanations_run():
    r = read('explanations_run').iloc[0]
    macro(['exp', 'seeds'], r.seeds, '{:.0f}')
    macro(['exp', 'jobs'], r.jobs, '{:.0f}')
    macro(['exp', 'failed'], r.failed, '{:.0f}')
    macro(['exp', 'explained', 'thousands'], r.explained/1e3, '{:.0f}')
    one = read('explanations_neighbourhoods')
    for _, r in one[one.family.isin(['all', 'smooth'])].iterrows():
        macro(['exp', r.setting, r.arm, r.kind, r.family, 'onesided'], pct(r.one_sided), '{:.0f}')


def numbers():
    lines = [f'\\newcommand{{\\{name}}}{{{value}}}' for name, value in sorted(NUMBERS.items())]
    write('numbers', '% written by make_tables.py from results/*.csv\n' + '\n'.join(lines) + '\n')
    listing = '\n'.join(f'{name} = {value}' for name, value in sorted(NUMBERS.items()))
    paths.out(paths.RESULTS, 'numbers.txt').write_text(listing + '\n')
    print(f'{len(NUMBERS)} numbers')


if __name__ == '__main__':
    for make in (datasets, replication, triad, arms, neighbourhoods, by_class, matched,
                 surrogate_thresholds, kernel, sweep, surrogate_deltas, runs,
                 failures, outliers, reproduce, explanations, explanations_paired,
                 explanations_moves, explanations_spread, explanations_instruments,
                 explanations_cuts, logistic_check, logistic_samples,
                 images, images_paired, images_samples, images_moves, images_models,
                 explanations_run):
        try:
            make()
        except (AttributeError, TypeError, FileNotFoundError) as e:   # a results file is not there yet
            print(f'skipped {make.__name__}: {e}')
    numbers()
