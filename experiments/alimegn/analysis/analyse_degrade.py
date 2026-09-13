'''
P3, P4, P6 from results_degrade.json

P3  as the black box degrades, yhat-derived class weights beat y-derived ones
P4  the ordering reverses when the target is the true labels rather than the black box
P6  degradation and the marginal mismatch are independent axes

P6 is measured with the variation along the line (max - min), not the boundary-to-tail
drop: see the note at the top of analyse_marginal.py for why the latter reads ~0.

usage:  uv run python analysis/analyse_degrade.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import re
import numpy as np
from scipy.stats import spearmanr, wilcoxon

import common_analysis as ca

NORMAL = 'bLIMEy (normal)'
CIKM = 'bLIMEy (cost sensitive sampled)'
GLOBAL_Y = 'bLIMEy (cost sensitive class)'
LOCAL_Y = 'bLIMEy (local y)'
LOCAL_YHAT = 'bLIMEy (local yhat)'
RATIO = 'bLIMEy (density ratio)'
SCHEMES = (CIKM, GLOBAL_Y, LOCAL_Y, LOCAL_YHAT, RATIO)


def divergence(entry):
    '''
    how far P(yhat|x) actually ended up from P(y|x), measured not assumed

    The locality-weighted rate at which the black box disagrees with the true label near
    the query points, averaged along the line.
    '''
    rows = entry.get('diagnostics', {}).get('local disagreement')
    if not rows:
        return float('nan')
    return float(np.nanmean([np.nan if r is None else r for r in rows]))


def mechanism(entry):
    model = entry['model']
    if entry.get('rebalancing', 'none') != 'none':
        return 'imbalance'
    if 'label noise' in model:
        return 'label noise'
    if any(tag in model for tag in ('underfit', 'stump', 'over-regularised', 'k=1')):
        return 'underfit'
    return 'clean'


def noise_rate(entry):
    found = re.search(r'label noise ([0-9.]+)', entry['model'])
    return float(found.group(1)) if found else 0.0


def base_model(entry):
    return re.sub(r' \((label noise [0-9.]+|underfit|stump|over-regularised|k=1)\)', '',
                  entry['model'])


def p3(results):
    print('\n== P3: does yhat beat y as the black box degrades? ==')
    print('local yhat vs local y: same points, same mechanism, different label source')
    print('(KL on test data, log10 ratio, positive favours yhat)\n')

    print(f"{'mechanism':14s} {'n':>4s} {'divergence':>11s} {'KL gain':>26s} "
          f"{'fidelity gain':>26s}")
    for name in ('clean', 'label noise', 'underfit', 'imbalance'):
        entries = [e for e in results.values() if mechanism(e) == name]
        if not entries:
            continue
        kl = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)', 'test data')
              for e in entries]
        fid = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'fidelity (local)', 'test data')
               for e in entries]
        div = np.nanmedian([divergence(e) for e in entries])
        print(f'{name:14s} {len(entries):4d} {div:11.3f} '
              f'{ca.fmt(*ca.median_and_count(kl)):>26s} '
              f'{ca.fmt(*ca.median_and_count(fid)):>26s}')

    print('\nalong the label-noise ladder (all bases and datasets pooled)\n')
    print(f"{'noise':>6s} {'n':>4s} {'divergence':>11s} {'KL gain':>26s} "
          f"{'fidelity gain':>26s}")
    ladder = [e for e in results.values() if mechanism(e) in ('clean', 'label noise')
              and 'seed=' not in e['key']]
    rates = sorted({noise_rate(e) for e in ladder})
    for rate in rates:
        entries = [e for e in ladder if noise_rate(e) == rate]
        kl = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)', 'test data')
              for e in entries]
        fid = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'fidelity (local)', 'test data')
               for e in entries]
        div = np.nanmedian([divergence(e) for e in entries])
        print(f'{rate:6.2f} {len(entries):4d} {div:11.3f} '
              f'{ca.fmt(*ca.median_and_count(kl)):>26s} '
              f'{ca.fmt(*ca.median_and_count(fid)):>26s}')

    print('\nthe same ladder, split by black box (KL gain of yhat over y)\n')
    print(f"{'base model':16s} " + ' '.join(f'{r:>8.2f}' for r in rates))
    for base in sorted({base_model(e) for e in ladder}):
        cells = []
        for rate in rates:
            entries = [e for e in ladder
                       if noise_rate(e) == rate and base_model(e) == base]
            kl = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)',
                                 'test data') for e in entries]
            median, _, _ = ca.median_and_count(kl)
            cells.append(f'{median:+8.3f}')
        print(f'{base:16s} ' + ' '.join(cells))

    # the registered prediction is about the trend against the MEASURED divergence
    div, kl, fid = [], [], []
    for entry in results.values():
        d = divergence(entry)
        if not np.isfinite(d):
            continue
        div.append(d)
        kl.append(ca.paired_gain(entry, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)',
                                 'test data'))
        fid.append(ca.paired_gain(entry, LOCAL_YHAT, LOCAL_Y, 'fidelity (local)',
                                  'test data'))
    div, kl, fid = np.array(div), np.array(kl), np.array(fid)
    for name, gain in (('KL', kl), ('fidelity', fid)):
        ok = np.isfinite(div) & np.isfinite(gain)
        rho = spearmanr(div[ok], gain[ok])
        print(f'\nmeasured divergence vs {name} gain of yhat over y: '
              f'rho = {rho.statistic:+.3f}, p = {rho.pvalue:.2g}, n = {ok.sum()}')

    # label noise only: within the mechanism the prediction was designed around
    noise_only = np.array([mechanism(e) in ('clean', 'label noise')
                           for e in results.values()])
    ok = np.isfinite(div) & np.isfinite(kl) & noise_only
    rho = spearmanr(div[ok], kl[ok])
    print(f'label noise only: rho = {rho.statistic:+.3f}, p = {rho.pvalue:.2g}, '
          f'n = {ok.sum()}')


def p4(results):
    print('\n== P4: does the ordering reverse when the target is the truth? ==')
    print('local y vs local yhat, scored on agreement with the TRUE test labels')
    print('(positive favours y, i.e. the reversal P4 predicts)\n')
    print(f"{'mechanism':14s} {'n':>4s} {'agreement with truth':>26s} "
          f"{'fidelity to f':>26s}")
    for name in ('clean', 'label noise', 'underfit', 'imbalance'):
        entries = [e for e in results.values() if mechanism(e) == name]
        if not entries:
            continue
        truth = [ca.paired_gain(e, LOCAL_Y, LOCAL_YHAT, 'surrogate vs truth')
                 for e in entries]
        fid = [ca.paired_gain(e, LOCAL_Y, LOCAL_YHAT, 'fidelity (local)', 'test data')
               for e in entries]
        print(f'{name:14s} {len(entries):4d} '
              f'{ca.fmt(*ca.median_and_count(truth)):>26s} '
              f'{ca.fmt(*ca.median_and_count(fid)):>26s}')

    degraded = [e for e in results.values() if mechanism(e) != 'clean']
    truth = np.array([ca.paired_gain(e, LOCAL_Y, LOCAL_YHAT, 'surrogate vs truth')
                      for e in degraded])
    fid = np.array([ca.paired_gain(e, LOCAL_Y, LOCAL_YHAT, 'fidelity (local)', 'test data')
                    for e in degraded])
    ok = np.isfinite(truth) & np.isfinite(fid)
    nonzero = ok & (truth != 0)
    if nonzero.sum() > 10:
        stat = wilcoxon(truth[nonzero])
        print(f'\nover the {ok.sum()} degraded configurations: y-weights better on truth '
              f'in {(truth[ok] > 0).sum()}, worse in {(truth[ok] < 0).sum()}, '
              f'tied in {(truth[ok] == 0).sum()}')
        print(f'Wilcoxon on the {nonzero.sum()} non-tied: p = {stat.pvalue:.2g} '
              f'(median {np.median(truth[nonzero]):+.4f})')
        print(f'the two objectives disagree in sign in '
              f'{((truth[ok] > 0) & (fid[ok] < 0)).sum()}/{ok.sum()} configurations')

    # how much does the choice of weighting matter for the truth objective at all?
    spread = []
    for entry in results.values():
        scores = [np.nanmean(ca.series(entry, s, 'surrogate vs truth'))
                  for s in (NORMAL,) + SCHEMES]
        scores = [s for s in scores if np.isfinite(s)]
        if len(scores) > 1:
            spread.append(max(scores) - min(scores))
    print(f'\nspread across all six schemes in agreement-with-truth: '
          f'median {np.median(spread):.4f}, 90th percentile '
          f'{np.percentile(spread, 90):.4f}')


def p6(results):
    print('\n== P6: are degradation and the marginal mismatch independent? ==')
    print("standard LIME's variation along the line, by noise rate\n")
    print(f"{'noise':>6s} {'n':>4s} {'variation (test)':>17s} {'variation (local)':>18s} "
          f"{'ratio':>7s} {'CIKM gain (test)':>18s}")
    ladder = [e for e in results.values() if mechanism(e) in ('clean', 'label noise')
              and 'seed=' not in e['key']]
    rates = sorted({noise_rate(e) for e in ladder})
    trend_x, trend_y = [], []
    for rate in rates:
        entries = [e for e in ladder if noise_rate(e) == rate]
        test = np.nanmedian([ca.variation(e, NORMAL, 'fidelity (local)', 'test data')
                             for e in entries])
        local = np.nanmedian([ca.variation(e, NORMAL, 'fidelity (local)', 'sample locally')
                              for e in entries])
        gain = np.nanmedian([ca.paired_gain(e, CIKM, NORMAL, 'fidelity (local)',
                                            'test data') for e in entries])
        print(f'{rate:6.2f} {len(entries):4d} {test:17.4f} {local:18.4f} '
              f'{test/local if local else np.nan:7.2f} {gain:18.4f}')
        for e in entries:
            v = ca.variation(e, NORMAL, 'fidelity (local)', 'test data')
            if np.isfinite(v):
                trend_x.append(noise_rate(e))
                trend_y.append(v)
    rho = spearmanr(trend_x, trend_y)
    print(f'\nnoise rate vs the size of the marginal effect: rho = {rho.statistic:+.3f}, '
          f'p = {rho.pvalue:.2g}, n = {len(trend_x)} '
          f'(P6 predicts no trend)')


def schemes_overall(results):
    print('\n== every scheme against standard LIME ==')
    print('(KL on test data, log10 ratio, positive favours the scheme)\n')
    degraded = [e for e in results.values() if mechanism(e) != 'clean']
    clean = [e for e in results.values() if mechanism(e) == 'clean']
    for scheme in SCHEMES:
        rows = []
        for entries in (clean, degraded):
            gains = [ca.paired_gain(e, scheme, NORMAL, 'KL divergence (local)', 'test data')
                     for e in entries]
            rows.append(ca.fmt(*ca.median_and_count(gains)))
        print(f'  {scheme:34s} clean {rows[0]:>24s}   degraded {rows[1]:>24s}')


def seed_spread(results):
    if not any('seed=' in k for k in results):
        return
    print('\n== seed spread: is a near-zero yhat-vs-y difference inside seed noise? ==')
    groups = {}
    for key, entry in results.items():
        groups.setdefault(key.split('|seed=')[0], []).append(entry)
    rows = []
    for base, entries in sorted(groups.items()):
        gains = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, 'KL divergence (local)',
                                'test data') for e in entries]
        gains = [g for g in gains if np.isfinite(g)]
        if len(gains) < 2:
            continue
        rows.append((base, len(gains), float(np.median(gains)), min(gains), max(gains),
                     np.sign(min(gains)) != np.sign(max(gains))))
    unstable = sum(1 for r in rows if r[5])
    print(f'{len(rows)} configurations run under 3 seeds; the sign of the yhat-vs-y '
          f'difference is unstable in {unstable} of them\n')
    print(f"{'configuration':52s} {'n':>3s} {'median':>9s} {'min':>9s} {'max':>9s}  sign")
    for base, n, med, lo, hi, is_unstable in rows:
        print(f'{base:52s} {n:3d} {med:+9.3f} {lo:+9.3f} {hi:+9.3f}  '
              f'{"UNSTABLE" if is_unstable else "stable"}')


if __name__ == '__main__':
    results, meta, _ = ca.load('results_degrade.json')
    print(f'{len(results)} configurations')
    p3(results)
    p4(results)
    p6(results)
    schemes_overall(results)
    seed_spread(results)
