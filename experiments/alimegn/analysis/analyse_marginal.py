'''
P1, P2, P5 from results_marginal.json

P1  the CIKM'23 collapse is a property of the evaluation marginal
P2  class weighting is a correction to that mismatch
P5  an explicit density ratio does it better, and the class trick approximates it

A note on how P1 is measured. The obvious statistic - fidelity at the boundary minus
fidelity in the tails - reads ~0.0000 on most configurations and is the wrong instrument.
Two reasons, both visible in the curves: the far ends of the line lie outside the data,
where the black box is locally constant and any surrogate agrees with it trivially, so
fidelity returns towards 1 there; and the collapse is usually asymmetric - one side of the
boundary only - so averaging the two ends cancels it. What the prediction is actually about
is how far the score moves ALONG the line, which is max - min.

usage:  uv run python analysis/analyse_marginal.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

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

# a configuration "shows the collapse" if standard LIME's test-set fidelity moves this far
# along the line
COLLAPSE = 0.1


def _balance(entry):
    row = entry.get('diagnostics', {}).get('local yhat balance')
    if not row:
        return None
    return np.array([np.nan if v is None else v for v in row], dtype=np.float64)


def p1(results):
    print('\n== P1: is the collapse a property of the evaluation marginal? ==')
    print('how far standard LIME\'s score moves along the line (max - min)\n')
    print(f"{'metric':22s} {'test data':>12s} {'own marginal':>14s} {'ratio':>7s} "
          f"{'bigger on test':>16s}")
    for metric in ('fidelity (local)', 'Brier score (local)', 'KL divergence (local)'):
        test = np.array([ca.variation(e, NORMAL, metric, 'test data')
                         for e in results.values()])
        local = np.array([ca.variation(e, NORMAL, metric, 'sample locally')
                          for e in results.values()])
        ok = np.isfinite(test) & np.isfinite(local)
        ratio = np.median(test[ok])/np.median(local[ok]) if np.median(local[ok]) else np.nan
        print(f'{metric:22s} {np.median(test[ok]):12.4f} {np.median(local[ok]):14.4f} '
              f'{ratio:7.2f} {f"{(test[ok] > local[ok]).sum()}/{ok.sum()}":>16s}')

    metric = 'fidelity (local)'
    test = np.array([ca.variation(e, NORMAL, metric, 'test data') for e in results.values()])
    local = np.array([ca.variation(e, NORMAL, metric, 'sample locally')
                      for e in results.values()])
    ok = np.isfinite(test) & np.isfinite(local)
    print(f'\nfidelity: Wilcoxon p = {wilcoxon(test[ok], local[ok]).pvalue:.2g}')

    worst_test = np.array([ca.worst_point(e, NORMAL, metric, 'test data')
                           for e in results.values()])
    worst_local = np.array([ca.worst_point(e, NORMAL, metric, 'sample locally')
                            for e in results.values()])
    print(f'worst query point: median fidelity {np.nanmedian(worst_test):.4f} on test data, '
          f'{np.nanmedian(worst_local):.4f} on its own marginal')

    # the conditional version: where the collapse happens at all, does it need the test set?
    shows = ok & (test > COLLAPSE)
    print(f'\nconfigurations whose test-set fidelity moves more than {COLLAPSE}: '
          f'{shows.sum()}/{ok.sum()}')
    print(f'  of those, median variation {np.median(test[shows]):.4f} on test data vs '
          f'{np.median(local[shows]):.4f} on the local sample '
          f'(ratio {np.median(test[shows])/np.median(local[shows]):.2f})')
    print(f'  smaller on the local sample in {(test[shows] > local[shows]).sum()}/'
          f'{shows.sum()}')
    collapse_local = ok & (local > COLLAPSE)
    print(f'configurations whose OWN-MARGINAL fidelity moves more than {COLLAPSE}: '
          f'{collapse_local.sum()}/{ok.sum()}')

    # the same surrogate, two distributions, point by point
    diffs = [ca.marginal_difference(e, NORMAL, metric) for e in results.values()]
    print(f'\nsame surrogate scored on its own marginal minus on test data: '
          f'{ca.fmt(*ca.median_and_count(diffs))}')

    # and the mechanism: the gap opens where the neighbourhood is one-sided
    onesided, gap = [], []
    for entry in results.values():
        balance = _balance(entry)
        if balance is None:
            continue
        local_series = ca.series(entry, NORMAL, metric, 'sample locally')
        test_series = ca.series(entry, NORMAL, metric, 'test data')
        onesided.extend(np.abs(balance - 0.5))
        gap.extend(local_series - test_series)
    onesided, gap = np.array(onesided), np.array(gap)
    ok = np.isfinite(onesided) & np.isfinite(gap)
    rho = spearmanr(onesided[ok], gap[ok])
    print(f'pooled over query points, how one-sided the neighbourhood is vs that gap: '
          f'rho = {rho.statistic:+.3f}, p = {rho.pvalue:.2g}, n = {ok.sum()}')


def p2(results):
    print('\n== P2: do class weights correct the marginal mismatch? ==')
    print('gain over standard LIME (fidelity: difference; KL: log10 ratio)\n')
    print(f"{'scheme':34s} {'metric':22s} {'test data':>24s} {'own marginal':>24s}")
    for scheme in SCHEMES:
        for metric in ('fidelity (local)', 'KL divergence (local)'):
            cells = []
            for eval_data in ('test data', 'sample locally'):
                gains = [ca.paired_gain(e, scheme, NORMAL, metric, eval_data)
                         for e in results.values()]
                cells.append(ca.fmt(*ca.median_and_count(gains)))
            print(f'{scheme:34s} {metric:22s} {cells[0]:>24s} {cells[1]:>24s}')

    print('\nand on the worst query point of each configuration (fidelity, test data)\n')
    for scheme in (NORMAL,) + SCHEMES:
        worst = [ca.worst_point(e, scheme, 'fidelity (local)', 'test data')
                 for e in results.values()]
        variation = [ca.variation(e, scheme, 'fidelity (local)', 'test data')
                     for e in results.values()]
        print(f'  {scheme:34s} worst point {np.nanmedian(worst):.4f}   '
              f'variation along the line {np.nanmedian(variation):.4f}')


def p5(results):
    print('\n== P5: the covariate-shift reading ==')
    print('density ratio against each other scheme, on test data '
          '(KL, log10 ratio, positive favours the density ratio)\n')
    for other in (NORMAL,) + SCHEMES[:-1]:
        gains = [ca.paired_gain(e, RATIO, other, 'KL divergence (local)', 'test data')
                 for e in results.values()]
        fid = [ca.paired_gain(e, RATIO, other, 'fidelity (local)', 'test data')
               for e in results.values()]
        print(f'  vs {other:34s} KL {ca.fmt(*ca.median_and_count(gains)):>24s}   '
              f'fidelity {ca.fmt(*ca.median_and_count(fid)):>24s}')

    print('\nagreement between CIKM class weights and the estimated density ratio')
    print('(Spearman over the 10,000 sampled points, per query point)\n')
    at_boundary, in_tails, everywhere = [], [], []
    for entry in results.values():
        agreement = entry.get('diagnostics', {}).get('weight agreement')
        if not agreement:
            continue
        values = np.array([np.nan if a is None else a for a in agreement])
        everywhere.extend(values[np.isfinite(values)])
        at_boundary.append(values[ca.boundary_index(entry)])
        in_tails.extend(values[ca.tail_mask(entry)])
    for name, values in (('all query points', everywhere),
                         ('at the boundary', at_boundary),
                         ('in the tails', in_tails)):
        values = np.array([v for v in values if v is not None and np.isfinite(v)])
        print(f'  {name:20s} median rho = {np.median(values):+.3f}  '
              f'({(values > 0.3).mean():.0%} above 0.3, n = {len(values)})')

    # per query point, not per configuration: a configuration-level average of
    # one-sidedness measures how well separated the classes are, which is a different
    # thing and correlates the other way (CIKM's own Finding 4)
    print('\nwhere does class weighting actually gain? pooled over query points\n')
    for scheme in (CIKM, RATIO):
        onesided, gain = [], []
        for entry in results.values():
            balance = _balance(entry)
            if balance is None:
                continue
            a = ca.series(entry, scheme, 'fidelity (local)', 'test data')
            b = ca.series(entry, NORMAL, 'fidelity (local)', 'test data')
            onesided.extend(np.abs(balance - 0.5))
            gain.extend(a - b)
        onesided, gain = np.array(onesided), np.array(gain)
        ok = np.isfinite(onesided) & np.isfinite(gain)
        rho = spearmanr(onesided[ok], gain[ok])
        print(f'  {scheme:34s} one-sidedness vs gain: rho = {rho.statistic:+.3f}, '
              f'p = {rho.pvalue:.2g}, n = {ok.sum()}')

    # the same thing at configuration level, which is the confound worth naming
    onesided, gain = [], []
    for entry in results.values():
        balance = _balance(entry)
        if balance is None:
            continue
        onesided.append(float(np.nanmean(np.abs(balance - 0.5))))
        gain.append(ca.paired_gain(entry, CIKM, NORMAL, 'fidelity (local)', 'test data'))
    onesided, gain = np.array(onesided), np.array(gain)
    ok = np.isfinite(onesided) & np.isfinite(gain)
    rho = spearmanr(onesided[ok], gain[ok])
    print(f'  {"(per configuration instead)":34s} rho = {rho.statistic:+.3f}, '
          f'p = {rho.pvalue:.2g}, n = {ok.sum()}')


def by_model(results):
    print('\n== per black box ==')
    print(f"{'model':22s} {'variation (test)':>17s} {'variation (local)':>18s} "
          f"{'CIKM gain':>24s} {'ratio gain':>24s}")
    for model in sorted({e['model'] for e in results.values()}):
        entries = [e for e in results.values() if e['model'] == model]
        test = np.nanmedian([ca.variation(e, NORMAL, 'fidelity (local)', 'test data')
                             for e in entries])
        local = np.nanmedian([ca.variation(e, NORMAL, 'fidelity (local)', 'sample locally')
                              for e in entries])
        cikm = [ca.paired_gain(e, CIKM, NORMAL, 'fidelity (local)', 'test data')
                for e in entries]
        ratio = [ca.paired_gain(e, RATIO, NORMAL, 'fidelity (local)', 'test data')
                 for e in entries]
        print(f'{model:22s} {test:17.4f} {local:18.4f} '
              f'{ca.fmt(*ca.median_and_count(cikm)):>24s} '
              f'{ca.fmt(*ca.median_and_count(ratio)):>24s}')


def by_dataset(results):
    print('\n== the configurations with the largest collapse (test-set fidelity) ==')
    rows = [(ca.variation(e, NORMAL, 'fidelity (local)', 'test data'),
             ca.variation(e, NORMAL, 'fidelity (local)', 'sample locally'),
             ca.worst_point(e, NORMAL, 'fidelity (local)', 'test data'),
             ca.worst_point(e, CIKM, 'fidelity (local)', 'test data'), key)
            for key, e in results.items()]
    rows.sort(reverse=True)
    print(f"{'configuration':46s} {'variation':>10s} {'(local)':>9s} "
          f"{'worst':>7s} {'worst+CIKM':>11s}")
    for test, local, worst, worst_cikm, key in rows[:12]:
        print(f'{key:46s} {test:10.3f} {local:9.3f} {worst:7.3f} {worst_cikm:11.3f}')


if __name__ == '__main__':
    results, meta, _ = ca.load('results_marginal.json')
    print(f'{len(results)} configurations, {meta.get("local_eval_samples")} '
          f'local evaluation points per query point')
    p1(results)
    p2(results)
    p5(results)
    by_model(results)
    by_dataset(results)
