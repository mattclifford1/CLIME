'''
P7, P8 from results_balance.json  (FINDINGS.md E4)

P7  local class imbalance is the signal; global class imbalance is not
P8  balanced training shrinks the gain, for the same reason degradation did (P6)

usage:  uv run python analysis/analyse_balance.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from scipy.stats import wilcoxon

import common_analysis as ca
from common import paths

NORMAL = 'bLIMEy (normal)'
SAMPLED = 'bLIMEy (cost sensitive sampled)'
GLOBAL_Y = 'bLIMEy (cost sensitive class)'
LOCAL_Y = 'bLIMEy (local y)'
LOCAL_YHAT = 'bLIMEy (local yhat)'
RATIO = 'bLIMEy (density ratio)'
SCHEMES = (SAMPLED, GLOBAL_Y, LOCAL_Y, LOCAL_YHAT, RATIO)
FIDELITY = 'fidelity (local)'

BALANCED_TAG = 'balanced training'


def cell(entry):
    '''which of the four cells of the design this configuration sits in'''
    data = ('undersampled' if entry.get('rebalancing', 'none') != 'none' else 'natural')
    model = 'balanced' if BALANCED_TAG in entry['model'] else 'normal'
    return data, model


def base_model(entry):
    return entry['model'].replace(f' {BALANCED_TAG}', '')


def p7(results):
    print('\n== P7: is the signal local imbalance or global imbalance? ==')
    print('gain over standard LIME in local fidelity on test data, by cell\n')
    print(f"{'training data':14s} {'black box':10s} {'n':>4s} " +
          ' '.join(f'{s.replace("bLIMEy ", "")[:22]:>24s}' for s in (SAMPLED, GLOBAL_Y)))
    for data in ('natural', 'undersampled'):
        for model in ('normal', 'balanced'):
            entries = [e for e in results.values() if cell(e) == (data, model)]
            if not entries:
                continue
            cells = []
            for scheme in (SAMPLED, GLOBAL_Y):
                gains = [ca.paired_gain(e, scheme, NORMAL, FIDELITY, 'test data')
                         for e in entries]
                cells.append(ca.fmt(*ca.median_and_count(gains)))
            print(f'{data:14s} {model:10s} {len(entries):4d} ' +
                  ' '.join(f'{c:>24s}' for c in cells))

    print('\nhead to head: local (sampled, from yhat) against global (from y)\n')
    for data in ('natural', 'undersampled'):
        entries = [e for e in results.values() if cell(e)[0] == data]
        gains = [ca.paired_gain(e, SAMPLED, GLOBAL_Y, FIDELITY, 'test data')
                 for e in entries]
        median, better, n = ca.median_and_count(gains)
        values = np.array([g for g in gains if np.isfinite(g)])
        p = wilcoxon(values).pvalue if len(values) > 10 and np.any(values != 0) else np.nan
        print(f'  {data:14s} local better in {better}/{n}, median {median:+.4f}, '
              f'p = {p:.2g}')

    print('\nevery scheme, pooled over the whole sweep (test data)\n')
    for scheme in SCHEMES:
        gains = [ca.paired_gain(e, scheme, NORMAL, FIDELITY, 'test data')
                 for e in results.values()]
        worst = np.nanmedian([ca.worst_point(e, scheme, FIDELITY, 'test data')
                              for e in results.values()])
        print(f'  {scheme:34s} {ca.fmt(*ca.median_and_count(gains)):>24s}   '
              f'worst point {worst:.3f}')
    worst = np.nanmedian([ca.worst_point(e, NORMAL, FIDELITY, 'test data')
                          for e in results.values()])
    print(f'  {NORMAL:34s} {"":>24s}   worst point {worst:.3f}')


def p8(results):
    print('\n== P8: does balanced training shrink the gain? ==')
    print('paired within (dataset, model family, training data): the same black box '
          'trained normally, then with balanced class weights\n')
    pairs = {}
    for entry in results.values():
        key = (entry['dataset'], base_model(entry), cell(entry)[0])
        pairs.setdefault(key, {})[cell(entry)[1]] = entry
    print(f"{'training data':14s} {'model':16s} {'n':>4s} {'normal':>10s} "
          f"{'balanced':>10s} {'shrinks':>9s}")
    rows = {}
    for data in ('natural', 'undersampled'):
        for model in sorted({base_model(e) for e in results.values()}):
            both = [v for k, v in pairs.items()
                    if k[1] == model and k[2] == data and len(v) == 2]
            if not both:
                continue
            normal = [ca.paired_gain(v['normal'], SAMPLED, NORMAL, FIDELITY, 'test data')
                      for v in both]
            balanced = [ca.paired_gain(v['balanced'], SAMPLED, NORMAL, FIDELITY,
                                       'test data') for v in both]
            shrinks = sum(1 for a, b in zip(normal, balanced)
                          if np.isfinite(a) and np.isfinite(b) and b < a)
            rows[(data, model)] = (normal, balanced)
            print(f'{data:14s} {model:16s} {len(both):4d} {np.nanmedian(normal):10.4f} '
                  f'{np.nanmedian(balanced):10.4f} {shrinks:5d}/{len(both):<3d}')

    normal = np.array([v for pair in rows.values() for v in pair[0]])
    balanced = np.array([v for pair in rows.values() for v in pair[1]])
    ok = np.isfinite(normal) & np.isfinite(balanced)
    p = wilcoxon(normal[ok], balanced[ok]).pvalue
    print(f'\npooled: gain {np.median(normal[ok]):+.4f} against a normally trained black '
          f'box, {np.median(balanced[ok]):+.4f} against a balance-trained one')
    print(f'smaller under balanced training in {(balanced[ok] < normal[ok]).sum()}/'
          f'{ok.sum()}, Wilcoxon p = {p:.2g}')

    # the mechanism P8 rests on: balanced training should make the neighbourhood less
    # one-sided in the region where the correction pays
    for name, index in (('natural', 'natural'), ('undersampled', 'undersampled')):
        one_sided = {}
        for entry in results.values():
            if cell(entry)[0] != index:
                continue
            row = entry.get('diagnostics', {}).get('sample yhat balance')
            if not row:
                continue
            values = np.array([np.nan if v is None else v for v in row])
            one_sided.setdefault(cell(entry)[1], []).append(
                float(np.nanmean(np.abs(values - 0.5))))
        if len(one_sided) == 2:
            print(f'  {name:14s} mean one-sidedness of the sampled neighbourhood: '
                  f'normal {np.mean(one_sided["normal"]):.3f}, '
                  f'balanced {np.mean(one_sided["balanced"]):.3f}')


def table(results):
    '''the E4 answer as a small LaTeX table'''
    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \footnotesize',
             r'  \caption{E4: local or global class imbalance? Median gain in local '
             r'fidelity over standard LIME on test data, over $14$ datasets $\times$ $3$ '
             r'model families, with the number of configurations improved. Local imbalance '
             r'helps in every cell; global imbalance in none.}',
             r'  \label{tab:balance}',
             r'  \begin{tabular}{llrrr}',
             r'    \toprule',
             r'    training data & black box & $n$ & local ($\hat{y}$, sample) '
             r'& global ($y$, training set) \\',
             r'    \midrule']
    for data in ('natural', 'undersampled'):
        for model in ('normal', 'balanced'):
            entries = [e for e in results.values() if cell(e) == (data, model)]
            if not entries:
                continue
            cells = []
            for scheme in (SAMPLED, GLOBAL_Y):
                gains = [ca.paired_gain(e, scheme, NORMAL, FIDELITY, 'test data')
                         for e in entries]
                median, better, n = ca.median_and_count(gains)
                cells.append(f'${median:+.4f}$ ({better}/{n})')
            label = 'class 0 at 20\\%' if data == 'undersampled' else 'natural'
            lines.append(f'    {label} & {model} & {len(entries)} & '
                         f'{cells[0]} & {cells[1]} \\\\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    out = paths.table('balance.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('\nwritten', out)


if __name__ == '__main__':
    results, meta, _ = ca.load('results_balance.json')
    print(f'{len(results)} configurations')
    p7(results)
    p8(results)
    table(results)
