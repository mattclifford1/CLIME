'''
Shared statistics for the diagnostic analyses of the fifth registration.

Rows are (dataset, black box) configurations. The quantities compared as predictors of the
advantage (standard LIME's local Brier score over Logit-LIME's) are:

    gap        Δ = R²_logit − R²_p, as registered and as the paper reports it
    r2_logit   R²_logit alone
    r2_prob    R²_p alone

Two ways of deciding which configurations the diagnostic is defined for:

    'paper'    analyse.degenerate: gap not finite, or |gap| > 1. The only option for files
               written before the guarded R² existed (results_taxonomy/extended.json)
    'guarded'  the guarded R² (sweep.guarded_r2) is defined at ≥ MIN_DEFINED of the 20 query
               points, i.e. f is not constant to rounding over most of the neighbourhoods.
               The fifth registration's rule; uses r2_logit_guarded / r2_prob_guarded

Intervals are a cluster bootstrap over *datasets*: the 16 black boxes fitted to one dataset
share its geometry and are not independent draws, so resampling configurations would
understate the uncertainty, and a p-value computed as if n were the configuration count
(the paper's p = 3e-29) overstates it.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

MIN_DEFINED = 10
BRIER, KL = 'Brier score (local)', 'KL divergence (local)'
STD, LOGIT = 'bLIMEy (normal)', 'bLIMEy (logit)'
PREDICTORS = ['gap', 'r2_logit', 'r2_prob']
LABEL = {'gap': 'Δ = R²_logit − R²_p', 'r2_logit': 'R²_logit', 'r2_prob': 'R²_p',
         'sat': 'saturation'}


def load(path, meta=None):
    '''rows from a sweep.py-format results file (taxonomy, extended, full, seeds, ...)'''
    d = json.load(open(path))
    rows = []
    for key, v in d.items():
        if key.startswith('_') or 'error' in v:
            continue
        dataset, model = key.split('|')
        b, k, diag = v['metrics'][BRIER], v['metrics'][KL], v['diagnostic']
        row = dict(dataset=dataset, model=model, group=v.get('group', 'unassigned'),
                   adv=b[STD]['mean']/max(b[LOGIT]['mean'], 1e-30),
                   adv_kl=k[STD]['mean']/max(k[LOGIT]['mean'], 1e-30),
                   brier_std=b[STD]['mean'], brier_logit=b[LOGIT]['mean'],
                   gap=diag['gap'], r2_logit=diag['r2_logit'], r2_prob=diag['r2_prob'],
                   sat=diag['saturation'])
        if 'r2_logit_guarded' in diag:
            row.update(r2_logit_g=diag['r2_logit_guarded'], r2_prob_g=diag['r2_prob_guarded'],
                       n_def=min(diag['n_defined_logit'], diag['n_defined_prob']))
        if meta and dataset in meta:
            row.update({f'meta_{k}': v for k, v in meta[dataset].items()})
        rows.append(row)
    return rows


def paper_degenerate(r):
    return not np.isfinite(r['gap']) or abs(r['gap']) > 1


def usable(rows, rule='paper'):
    '''(rows the diagnostic is defined for, with predictors set by the rule; n excluded)'''
    if rule == 'paper':
        ok = [r for r in rows if not paper_degenerate(r)]
    elif rule == 'guarded':
        ok = []
        for r in rows:
            if r.get('n_def', 0) < MIN_DEFINED:
                continue
            ok.append({**r, 'r2_logit': r['r2_logit_g'], 'r2_prob': r['r2_prob_g'],
                       'gap': r['r2_logit_g'] - r['r2_prob_g']})
    else:
        raise ValueError(rule)
    return ok, len(rows) - len(ok)


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 else float('nan')


def partial_rho(x, y, z):
    '''rank correlation of x and y with z partialled out of both'''
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.c_[rz, np.ones_like(rz)]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def auc(score, adv, threshold=2.0):
    y = np.asarray(adv) > threshold
    if y.all() or not y.any():
        return float('nan')
    return float(roc_auc_score(y, score))


def precision_recall(flag, adv, threshold=2.0):
    flag, y = np.asarray(flag, bool), np.asarray(adv) > threshold
    prec = float(y[flag].mean()) if flag.any() else float('nan')
    rec = float(flag[y].mean()) if y.any() else float('nan')
    return prec, rec, int(flag.sum())


def summary(rows):
    '''every number the diagnostic comparison reports, for one set of rows'''
    col = {k: np.array([r[k] for r in rows], float) for k in PREDICTORS + ['adv', 'sat']}
    out = {'n': len(rows), 'n_datasets': len({r['dataset'] for r in rows})}
    for k in PREDICTORS + ['sat']:
        out[f'rho_{k}'] = rho(col[k], col['adv'])
        out[f'auc_{k}'] = auc(col[k], col['adv'])
        out[f'auc10_{k}'] = auc(col[k], col['adv'], 10)
    out['partial_gap|r2_logit'] = partial_rho(col['gap'], col['adv'], col['r2_logit'])
    out['partial_r2_logit|r2_prob'] = partial_rho(col['r2_logit'], col['adv'], col['r2_prob'])
    out['partial_r2_prob|r2_logit'] = partial_rho(col['r2_prob'], col['adv'], col['r2_logit'])
    out['rule_r2_logit>0.95'] = precision_recall(col['r2_logit'] > 0.95, col['adv'])
    out['rule_gap>0.35'] = precision_recall(col['gap'] > 0.35, col['adv'])
    return out


def cluster_bootstrap(rows, stat, B=2000, seed=0):
    '''95% percentile interval of stat(rows), resampling datasets with replacement'''
    by = {}
    for r in rows:
        by.setdefault(r['dataset'], []).append(r)
    names = sorted(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(len(names), len(names), replace=True)
        sample = [r for i in pick for r in by[names[i]]]
        v = stat(sample)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rho_stat(key, target='adv'):
    return lambda rs: rho([r[key] for r in rs], [r[target] for r in rs])


def auc_stat(key, threshold=2.0):
    return lambda rs: auc([r[key] for r in rs], [r['adv'] for r in rs], threshold)


def diff_stat(a, b):
    '''ρ(a, adv) − ρ(b, adv): is one predictor better than the other?'''
    return lambda rs: rho_stat(a)(rs) - rho_stat(b)(rs)
