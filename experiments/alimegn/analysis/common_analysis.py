'''
shared reading and aggregating of the sweep results

Every sweep here stores per-query-point lists, because the effects are about how a score
varies ALONG the line: a configuration mean would average P1 away. These helpers do the
three things every analysis script needs - load, pick out the boundary and the tails, and
compare two schemes with the right sign for the metric.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import paths   # noqa: E402

# lower is better for the divergences, higher is better for the agreements
HIGHER_IS_BETTER = {'fidelity (local)': True, 'Brier score (local)': False,
                    'KL divergence (local)': False, 'surrogate vs truth': True}

# how many query points at each end of the line count as "the tails"
TAIL_POINTS = 3


def load(name):
    with open(paths.results(name)) as f:
        results = json.load(f)
    meta = results.pop('_meta', {})
    good = {k: v for k, v in results.items() if 'error' not in v}
    bad = {k: v for k, v in results.items() if 'error' in v}
    if bad:
        print(f'# {len(bad)} configurations failed and are excluded:', flush=True)
        for k, v in list(bad.items())[:8]:
            print(f'#   {k:60s} {v["error"][:70]}')
    return good, meta, bad


def series(entry, scheme, metric, eval_data=None):
    '''the per-query-point scores as a float array, with None -> nan'''
    record = entry['schemes'][scheme]
    values = record[metric] if eval_data is None else record[eval_data][metric]
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def boundary_index(entry):
    '''
    the query point closest to the decision boundary

    Taken from the black box's own predictions on nearby test points (the `local yhat
    balance` diagnostic): the boundary is where the local neighbourhood is evenly split
    between the two predicted classes. Falls back to the middle of the line when the
    diagnostics were not recorded.
    '''
    balance = entry.get('diagnostics', {}).get('local yhat balance')
    if not balance:
        return entry['n_query_points']//2
    balance = np.array([np.nan if b is None else b for b in balance], dtype=np.float64)
    if np.all(np.isnan(balance)):
        return entry['n_query_points']//2
    return int(np.nanargmin(np.abs(balance - 0.5)))


def tail_mask(entry):
    n = entry['n_query_points']
    mask = np.zeros(n, dtype=bool)
    mask[:TAIL_POINTS] = True
    mask[-TAIL_POINTS:] = True
    return mask


def boundary_to_tail_drop(entry, scheme, metric, eval_data):
    '''
    how much worse the surrogate is in the tails than at the boundary

    This is the CIKM'23 effect as a single number, with the sign arranged so that positive
    always means "worse away from the boundary".
    '''
    values = series(entry, scheme, metric, eval_data)
    at_boundary = values[boundary_index(entry)]
    in_tails = np.nanmean(values[tail_mask(entry)])
    drop = at_boundary - in_tails
    return drop if HIGHER_IS_BETTER[metric] else -drop


def variation(entry, scheme, metric, eval_data):
    '''
    how much the score moves along the line: worst query point against best

    The boundary-to-tail drop turns out to be the wrong instrument for P1 (it reads ~0 on
    most configurations): the far ends of the line sit outside the data, where the black
    box and any surrogate agree trivially because both predict one class over the whole
    neighbourhood, so fidelity returns to ~1 there. What CIKM'23 reports is a collapse
    somewhere along the line, not specifically at its ends, and this measures that.
    '''
    values = series(entry, scheme, metric, eval_data)
    if np.all(np.isnan(values)):
        return float('nan')
    spread = np.nanmax(values) - np.nanmin(values)
    return float(spread)


def worst_point(entry, scheme, metric, eval_data):
    '''the score at the query point where the surrogate does worst'''
    values = series(entry, scheme, metric, eval_data)
    if np.all(np.isnan(values)):
        return float('nan')
    return float(np.nanmin(values) if HIGHER_IS_BETTER[metric] else np.nanmax(values))


def marginal_difference(entry, scheme, metric):
    '''
    how much better the surrogate looks when scored on its own training marginal

    Positive means the local sample flatters it, which is what P1 is about: the same
    surrogate judged against two different distributions.
    '''
    local = series(entry, scheme, metric, 'sample locally')
    test = series(entry, scheme, metric, 'test data')
    if HIGHER_IS_BETTER[metric]:
        return float(np.nanmean(local - test))
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nanmean(np.log10(np.maximum(test, 1e-30))
                                - np.log10(np.maximum(local, 1e-30))))


def paired_gain(entry, scheme, baseline, metric, eval_data=None):
    '''
    mean improvement of `scheme` over `baseline` across the query points

    Positive always favours `scheme`. For the divergences the comparison is made in the
    log, since Brier and KL span orders of magnitude and a difference of means would be
    decided entirely by the worst query point.
    '''
    a = series(entry, scheme, metric, eval_data)
    b = series(entry, baseline, metric, eval_data)
    if HIGHER_IS_BETTER[metric]:
        return float(np.nanmean(a - b))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log10(np.maximum(b, 1e-30)) - np.log10(np.maximum(a, 1e-30))
    return float(np.nanmean(ratio))   # >0 means `scheme` has the smaller divergence


def median_and_count(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(values) == 0:
        return float('nan'), 0, 0
    return float(np.median(values)), int((values > 0).sum()), len(values)


def fmt(median, better, n, unit=''):
    return f'{median:+.4f}{unit} (better in {better}/{n})'
