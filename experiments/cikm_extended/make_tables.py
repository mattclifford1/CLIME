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
        rows.append([f'\\emph{{{label}}}'] + ['']*8)
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
                             f'{t.wins:.0f}/{t.ties:.0f}/{t.losses:.0f}', f'{c.n:.0f}'])
        rows.append([])
    write('replication', table(['', 'LIME', 'CLIME', 'CLIME $-$ LIME', '+/=/$-$',
                                'Re-thr.', 'Re-thr.\\ $-$ LIME', '+/=/$-$', 'Blocks'],
                               rows, 'lrrrcrrcr'))


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
        base = ['bin', r.setting, r.kind, r.minority.replace('%', ' pct').replace('-', ' to '),
                MAIN[r.explainer].split(',')[-1]]
        macro([*base, 'fid'], r.test_fidelity)
        macro([*base, 'bal'], r.test_balanced)
        macro([*base, 'cosine'], r.cosine)


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


def reproduce():
    frame = read('reproduce')
    for (name, explainer), part in frame.groupby(['configuration', 'explainer']):
        macro(['pub', name.replace('±1', 'one').replace('±3', 'three'), explainer, 'mean'],
              part.fidelity.mean(), '{:.3f}')
        macro(['pub', name.replace('±1', 'one').replace('±3', 'three'), explainer, 'min'],
              part.fidelity.min())


def numbers():
    lines = [f'\\newcommand{{\\{name}}}{{{value}}}' for name, value in sorted(NUMBERS.items())]
    write('numbers', '% written by make_tables.py from results/*.csv\n' + '\n'.join(lines) + '\n')
    listing = '\n'.join(f'{name} = {value}' for name, value in sorted(NUMBERS.items()))
    paths.out(paths.RESULTS, 'numbers.txt').write_text(listing + '\n')
    print(f'{len(NUMBERS)} numbers')


if __name__ == '__main__':
    for make in (datasets, replication, triad, arms, neighbourhoods, surrogate_thresholds,
                 reproduce):
        try:
            make()
        except (AttributeError, TypeError) as e:   # a results file is not there yet
            print(f'skipped {make.__name__}: {e}')
    numbers()
