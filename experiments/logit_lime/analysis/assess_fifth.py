'''
Score every prediction of the fifth registration (PREREGISTRATION.md) against the full-grid
results, and write results/assessment_fifth.json.

Each check prints its registered threshold, the measured value and HELD / FAILED. Anything
whose results file is not there yet is reported as MISSING rather than skipped silently.

usage:  python analysis/assess_fifth.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import json
import numpy as np
from analysis import diagnostic_stats as ds
from sweeps import full_grid

GROUP_A = ['Logistic', 'LDA', 'Nearest Class Mean']
GROUP_D = ['Decision Tree', 'Random Forest', 'k Nearest Neighbours']
GEOMETRY_FOLLOWERS = ['QDA', 'Gaussian Naive Bayes', 'Bayes Optimal']

OUT = {}


def verdict(name, registered, measured, held, note=''):
    tag = 'HELD' if held else 'FAILED'
    print(f'{name:4s} {tag:7s} registered: {registered:46s} measured: {measured}'
          + (f'   ({note})' if note else ''))
    OUT[name] = {'registered': registered, 'measured': measured, 'held': bool(held),
                 'note': note}


def missing(name, what):
    print(f'{name:4s} MISSING {what}')
    OUT[name] = {'missing': what}


def exists(f):
    return os.path.exists(paths.results(f))


def load_json(f):
    return json.load(open(paths.results(f)))


def adv_lookup(rows):
    return {(r['dataset'], r['model']): r['adv'] for r in rows}


def main():
    meta = load_json('dataset_meta.json') if exists('dataset_meta.json') else None
    if not exists('results_full.json'):
        missing('all', 'results_full.json')
        return
    full = ds.load(paths.results('results_full.json'), meta)
    new = [r for r in full if r['dataset'] in full_grid.NEW]
    new_ok, new_excl = ds.usable(new, 'guarded')
    print(f'new configurations: {len(new)} ({len(full_grid.NEW)} datasets x 16 = '
          f'{16*len(full_grid.NEW)} expected), {new_excl} excluded by the guarded rule\n')

    # N1, N2
    s = ds.summary(new_ok)
    verdict('N1', 'rho(R2logit) > rho(gap); AUC(R2logit) >= .95 and > gap',
            f"rho {s['rho_r2_logit']:+.3f} vs {s['rho_gap']:+.3f}; "
            f"AUC {s['auc_r2_logit']:.3f} vs {s['auc_gap']:.3f}",
            s['rho_r2_logit'] > s['rho_gap'] and s['auc_r2_logit'] >= 0.95
            and s['auc_r2_logit'] > s['auc_gap'], f'n = {s["n"]}')
    p, r, n = s['rule_r2_logit>0.95']
    verdict('N2', 'R2logit > .95 => adv > 2x, precision >= .85',
            f'precision {p:.3f}, recall {r:.3f}, {n} flagged', p >= 0.85)

    # N3, N4 - the advantage is defined whether or not the diagnostic is
    a = [r for r in new if r['model'] in GROUP_A]
    wins = sum(r['adv'] > 1 for r in a)
    losers = sorted((r['adv'], r['dataset'], r['model']) for r in a if r['adv'] <= 1)
    verdict('N3', 'group A better on >= 124 of 126', f'{wins} of {len(a)}', wins >= 124,
            'worst: ' + '; '.join(f'{d}|{m} {v:.3g}x' for v, d, m in losers[:5]))
    dd = [r['adv'] for r in new if r['model'] in GROUP_D]
    verdict('N4', 'group D median advantage < 1', f'{np.median(dd):.3f} (n = {len(dd)})',
            np.median(dd) < 1)

    # N5 - Bayes Optimal is not in the 16-model grid; its advantage comes from the
    # fidelity sweep's (Brier, local sample) cell, which reproduces the main protocol
    advs = adv_lookup(full)
    if exists('results_fidelity_full.json'):
        fid = load_json('results_fidelity_full.json')
        for k, v in fid.items():
            if k.startswith('_') or 'error' in v:
                continue
            c = v['cells']['Brier | local sample']
            advs.setdefault(tuple(k.split('|')), c[ds.STD]['mean']/max(c[ds.LOGIT]['mean'], 1e-30))
    cells, detail = 0, []
    for d in (2, 5, 10, 30):
        for sep in (2, 4, 6):
            med = {}
            for r_ in (1, 3):
                vals = [advs.get((f'Gauss d{d} r{r_} s{sep}', m)) for m in GEOMETRY_FOLLOWERS]
                vals = [v for v in vals if v is not None]
                med[r_] = np.median(vals) if vals else np.nan
            cells += med[1] > med[3]
            detail.append(f'd{d}s{sep}: {med[1]:.3g}/{med[3]:.3g}')
    verdict('N5', 'median adv r=1 > r=3 in >= 10 of 12 (d, s) cells', f'{cells} of 12',
            cells >= 10, ', '.join(detail))

    # N6
    g_ok, g_excl = ds.usable([r for r in full if full_grid.FAMILY_OF[r['dataset']] == 'gaussian'],
                             'guarded')
    gs = ds.summary(g_ok)
    verdict('N6', 'Gaussian family rho(R2logit) >= .6, |rho(sat)| < .3',
            f"rho(R2logit) {gs['rho_r2_logit']:+.3f}, rho(sat) {gs['rho_sat']:+.3f}",
            gs['rho_r2_logit'] >= 0.6 and abs(gs['rho_sat']) < 0.3,
            f'n = {gs["n"]}, {g_excl} excluded')

    # N7 - reported, not tested
    rows = []
    for sep in (2, 4, 6):
        vals = [advs[(f'Gauss d{d} r{r_} s{sep}', m)] for d in (2, 5, 10, 30)
                for r_ in (1, 3) for m in GROUP_A if (f'Gauss d{d} r{r_} s{sep}', m) in advs]
        rows.append(f's={sep}: median {np.median(vals):.3g}x')
    print(f"N7   REPORT  group A advantage by separation: {', '.join(rows)}")
    OUT['N7'] = {'report': rows}

    # N8
    if exists('results_gradient_truth_full.json'):
        gt = load_json('results_gradient_truth_full.json')
        per_d = {}
        for name in full_grid.MAKECLF:
            d, _ = full_grid.makeclf_params(name)
            for m in GROUP_A:
                e = gt.get(f'{name}|{m}')
                if e and 'error' not in e:
                    per_d.setdefault(d, []).append((e['top1_logit'], e['top1_standard']))
        mins = {d: (min(v[0] for v in vs), min(v[1] for v in vs)) for d, vs in sorted(per_d.items())}
        verdict('N8', 'MakeClf group A: top-1 logit >= .9, standard >= .5, every d',
                ', '.join(f'd{d}: {l:.2f}/{s_:.2f}' for d, (l, s_) in mins.items()),
                all(l >= 0.9 and s_ >= 0.5 for l, s_ in mins.values()),
                'minimum over datasets and group A models, logit/standard')
    else:
        missing('N8', 'results_gradient_truth_full.json')

    # C1-C5
    if exists('results_diagnostic_checks.json'):
        ck = load_json('results_diagnostic_checks.json')
        crow = [dict(v, dataset=k.split('|')[0], model=k.split('|')[1]) for k, v in ck.items()
                if not k.startswith('_') and 'error' not in v
                and v.get('n_defined_r2_logit', 0) >= ds.MIN_DEFINED]
        for r in crow:
            r['adv'] = advs.get((r['dataset'], r['model']), np.nan)
        col = lambda k: np.array([r.get(k, np.nan) for r in crow], float)
        adv = col('adv')
        r13 = ds.rho(col('r2_logit_eps1e-03'), col('r2_logit_eps1e-12'))
        per_eps = {e: ds.rho(col(f'r2_logit_eps{e:.0e}'), adv) for e in (1e-3, 1e-6, 1e-9, 1e-12)}
        spread = max(per_eps.values()) - min(per_eps.values())
        verdict('C1', 'rank corr eps 1e-3 vs 1e-12 >= .9; rho(adv) moves < .05',
                f'{r13:+.3f}; rho by eps ' + ', '.join(f'{e:.0e}: {v:+.3f}'
                                                       for e, v in per_eps.items()),
                r13 >= 0.9 and spread < 0.05, f'n = {len(crow)}, spread {spread:.3f}')
        ga = [r['curvature_gain'] for r in crow if r['model'] in GROUP_A]
        rest = [r for r in crow if r['model'] not in GROUP_A]
        rc = ds.rho([r['curvature_gain'] for r in rest], [r['adv'] for r in rest])
        verdict('C2', 'group A median curvature gain < .01; rho(gain, adv) <= -.3 outside A',
                f'{np.median(ga):.4f}; rho {rc:+.3f}', np.median(ga) < 0.01 and rc <= -0.3)
        disp = [r for r in crow if np.isfinite(r.get('grad_dispersion', np.nan))]
        a_disp = max(r['grad_dispersion'] for r in disp if r['model'] in GROUP_A)
        rd = ds.rho([r['grad_dispersion'] for r in disp], [r['r2_logit'] for r in disp])
        verdict('C3', 'group A dispersion 0; rho(dispersion, R2logit) <= -.6',
                f'max group A {a_disp:.2g}; rho {rd:+.3f}', a_disp < 1e-12 and rd <= -0.6,
                f'n = {len(disp)}')
        ins = col('insample_brier_prob')/np.maximum(col('insample_brier_logit'), 1e-30)
        r_ins, r_r2 = ds.rho(ins, adv), ds.rho(col('r2_logit'), adv)
        verdict('C4', 'rho(in-sample ratio, adv) >= .9 and > rho(R2logit)',
                f'{r_ins:+.3f} vs {r_r2:+.3f}', r_ins >= 0.9 and r_ins > r_r2)
        r5 = ds.rho(col('own_r2_logit'), col('r2_logit'))
        verdict('C5', 'rank corr own-salt vs evaluation-salt R2logit >= .98', f'{r5:+.4f}',
                r5 >= 0.98)
        OUT['_checks_n'] = len(crow)
    else:
        for c in ('C1', 'C2', 'C3', 'C4', 'C5'):
            missing(c, 'results_diagnostic_checks.json')

    # S
    base = ds.rho_stat('r2_logit')(ds.usable(full, 'guarded')[0])
    per_seed = {42: base}
    for sd in (1, 2, 3, 4):
        f = f'results_full_seed{sd}.json'
        if exists(f):
            per_seed[sd] = ds.rho_stat('r2_logit')(ds.usable(ds.load(paths.results(f)), 'guarded')[0])
    if len(per_seed) == 5:
        verdict('S', 'rho >= .7 every seed, within .05 of seed 42',
                ', '.join(f'{k}: {v:+.3f}' for k, v in per_seed.items()),
                all(v >= 0.7 and abs(v - base) <= 0.05 for v in per_seed.values()))
    else:
        missing('S', f'seed files ({len(per_seed)} of 5 present)')

    # K
    if exists('results_kernel_full.json'):
        kr = load_json('results_kernel_full.json')
        per_scale = {}
        for k, v in kr.items():
            if k.startswith('_') or 'error' in v:
                continue
            scale, dataset, model = k.split('|')
            dg = v['diagnostic']
            if min(dg.get('n_defined_logit', 0), dg.get('n_defined_prob', 0)) < ds.MIN_DEFINED:
                continue
            per_scale.setdefault(float(scale), []).append(
                (dg['r2_logit_guarded'], v[ds.STD]['mean']/max(v[ds.LOGIT]['mean'], 1e-30)))
        rhos = {s_: ds.rho([a for a, _ in v], [b for _, b in v]) for s_, v in sorted(per_scale.items())}
        verdict('K', 'rho >= .6 at every scale >= .3',
                ', '.join(f'{s_}: {v:+.3f}' for s_, v in rhos.items()),
                all(v >= 0.6 for s_, v in rhos.items() if s_ >= 0.3))
        OUT['K']['per_scale'] = rhos
    else:
        missing('K', 'results_kernel_full.json')

    # Q
    if exists('results_querypoints_full.json'):
        qr = ds.load(paths.results('results_querypoints_full.json'))
        qa = [r for r in qr if r['model'] in GROUP_A]
        frac = np.mean([r['adv'] > 1 for r in qa])
        qrho = ds.rho_stat('r2_logit')(ds.usable(qr, 'guarded')[0])
        verdict('Q', 'group A better >= 95%; rho >= .6',
                f'{frac:.1%} of {len(qa)}; rho {qrho:+.3f}', frac >= 0.95 and qrho >= 0.6)
    else:
        missing('Q', 'results_querypoints_full.json')

    # R
    if exists('results_ridge_alpha.json'):
        ra = load_json('results_ridge_alpha.json')
        same, tot = 0, 0
        for k, v in ra.items():
            if k.startswith('_') or 'error' in v:
                continue
            a_ = v[f'advantage {ds.BRIER}']
            signs = {np.sign(np.log(a_[x])) for x in ('0.1', '1.0', '10.0')}
            tot += 1
            same += len(signs) == 1
        verdict('R', 'sign of log(adv) same at alpha .1, 1, 10 in >= 95%',
                f'{same} of {tot} ({same/max(tot,1):.1%})', same/max(tot, 1) >= 0.95)
    else:
        missing('R', 'results_ridge_alpha.json')

    json.dump(OUT, open(paths.results('assessment_fifth.json'), 'w'), indent=1, default=float)
    print('\nwritten', paths.results('assessment_fifth.json'))


if __name__ == '__main__':
    main()
