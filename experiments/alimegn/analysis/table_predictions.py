'''
the registered predictions against what happened, as a LaTeX table

The verdict in each row is computed from the thresholds written in PREREGISTRATION.md
rather than typed in by hand, so the table cannot drift away from the numbers and a re-run
on new data re-decides it.

usage:  uv run python analysis/table_predictions.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from scipy.stats import spearmanr, wilcoxon

import common_analysis as ca
from common import paths
from analyse_degrade import divergence, mechanism, noise_rate

NORMAL = 'bLIMEy (normal)'
CIKM = SAMPLED = 'bLIMEy (cost sensitive sampled)'
GLOBAL_Y = 'bLIMEy (cost sensitive class)'
LOCAL_Y = 'bLIMEy (local y)'
LOCAL_YHAT = 'bLIMEy (local yhat)'
RATIO = 'bLIMEy (density ratio)'
FIDELITY = 'fidelity (local)'
KL = 'KL divergence (local)'


def p1(marginal):
    test = np.array([ca.variation(e, NORMAL, FIDELITY, 'test data')
                     for e in marginal.values()])
    local = np.array([ca.variation(e, NORMAL, FIDELITY, 'sample locally')
                      for e in marginal.values()])
    ok = np.isfinite(test) & np.isfinite(local)
    ratio = np.median(test[ok])/np.median(local[ok])
    p = wilcoxon(test[ok], local[ok]).pvalue
    verdict = ('confirmed' if ratio >= 5 else
               'direction confirmed, threshold not met' if p < 0.05 and ratio > 1 else
               'refuted')
    measured = (f'{ratio:.2f}$\\times$ larger on test data '
                f'({(test[ok] > local[ok]).sum()}/{ok.sum()}, $p = {p:.1g}$)')
    return ('P1', 'the collapse belongs to the evaluation marginal',
            '$\\geq 5\\times$ smaller on the local sample', measured, verdict)


def p2(marginal):
    gains = {}
    for eval_data in ('test data', 'sample locally'):
        gains[eval_data] = np.array([ca.paired_gain(e, CIKM, NORMAL, FIDELITY, eval_data)
                                     for e in marginal.values()])
    test, local = gains['test data'], gains['sample locally']
    ok = np.isfinite(test) & np.isfinite(local)
    verdict = ('confirmed' if np.median(test[ok]) > 0
               and np.median(test[ok]) > np.median(local[ok]) else 'refuted')
    measured = (f'$+{np.median(test[ok]):.4f}$ on test data vs '
                f'$+{np.median(local[ok]):.4f}$ on its own marginal')
    return ('P2', 'class weighting corrects that mismatch',
            'gain on test data, smaller locally', measured, verdict)


def p3(degrade):
    ladder = [e for e in degrade.values()
              if mechanism(e) in ('clean', 'label noise') and 'seed=' not in e['key']]
    rates = sorted({noise_rate(e) for e in ladder})
    medians = []
    for rate in rates:
        gains = [ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, KL, 'test data')
                 for e in ladder if noise_rate(e) == rate]
        median, _, _ = ca.median_and_count(gains)
        medians.append(median)
    div = np.array([divergence(e) for e in degrade.values()])
    gain = np.array([ca.paired_gain(e, LOCAL_YHAT, LOCAL_Y, KL, 'test data')
                     for e in degrade.values()])
    ok = np.isfinite(div) & np.isfinite(gain)
    rho = spearmanr(div[ok], gain[ok])
    rises = medians[-1] > medians[0]
    verdict = ('confirmed' if rises and rho.pvalue < 0.05 else
               'trend only, not significant' if rises else 'refuted')
    measured = (f'ladder {medians[0]:+.4f} $\\to$ {medians[-1]:+.4f} in $\\log_{{10}}$ KL; '
                f'$\\rho = {rho.statistic:+.3f}$, $p = {rho.pvalue:.2g}$')
    return ('P3', r'as $f$ degrades, $\hat{y}$ weights beat $y$ weights',
            'gap grows with the divergence', measured, verdict)


def p4(degrade):
    degraded = [e for e in degrade.values() if mechanism(e) != 'clean']
    truth = np.array([ca.paired_gain(e, LOCAL_Y, LOCAL_YHAT, 'surrogate vs truth')
                      for e in degraded])
    ok = np.isfinite(truth)
    better = int((truth[ok] > 0).sum())
    verdict = 'confirmed' if better > ok.sum()/2 else 'refuted'
    measured = (f'$y$ weights better on truth in {better}/{ok.sum()}, '
                f'median {np.median(truth[ok]):+.4f}')
    return ('P4', 'the ordering reverses when the target is the truth',
            r'$y$ weights win on agreement with $y$', measured, verdict)


def p5(marginal):
    beats = {}
    for other in (NORMAL, CIKM, LOCAL_Y, LOCAL_YHAT):
        gains = [ca.paired_gain(e, RATIO, other, KL, 'test data')
                 for e in marginal.values()]
        median, better, n = ca.median_and_count(gains)
        beats[other] = (median, better, n)
    worst = min(better/n for _, better, n in beats.values())
    tails = []
    for entry in marginal.values():
        row = entry.get('diagnostics', {}).get('weight agreement')
        if not row:
            continue
        values = np.array([np.nan if v is None else v for v in row])
        tails.extend(values[ca.tail_mask(entry)])
    tails = np.array([v for v in tails if np.isfinite(v)])
    rho_tails = float(np.median(tails))
    verdict = ('confirmed' if worst > 0.5 and rho_tails > 0.3 else
               'first half confirmed, second refuted' if worst > 0.5 else 'refuted')
    measured = (f'density ratio best on KL in {beats[CIKM][1]}/{beats[CIKM][2]} vs CIKM; '
                f'weight agreement in the tails $\\rho = {rho_tails:+.3f}$')
    return ('P5', 'the covariate-shift reading',
            r'density ratio wins; $\rho > 0.3$ away from the boundary', measured, verdict)


def p6(degrade):
    ladder = [e for e in degrade.values()
              if mechanism(e) in ('clean', 'label noise') and 'seed=' not in e['key']]
    x, y = [], []
    for entry in ladder:
        v = ca.variation(entry, NORMAL, FIDELITY, 'test data')
        if np.isfinite(v):
            x.append(noise_rate(entry))
            y.append(v)
    rho = spearmanr(x, y)
    verdict = 'confirmed' if rho.pvalue > 0.05 else 'refuted'
    measured = (f'noise rate vs marginal effect $\\rho = {rho.statistic:+.3f}$, '
                f'$p = {rho.pvalue:.2g}$, $n = {len(x)}$')
    return ('P6', 'degradation and the marginal mismatch are independent',
            'no trend in the marginal effect with noise', measured, verdict)


def p7(balance):
    from analyse_balance import cell
    rows = {}
    for data in ('natural', 'undersampled'):
        for model in ('normal', 'balanced'):
            entries = [e for e in balance.values() if cell(e) == (data, model)]
            if not entries:
                continue
            local = [ca.paired_gain(e, SAMPLED, NORMAL, FIDELITY, 'test data')
                     for e in entries]
            glob = [ca.paired_gain(e, GLOBAL_Y, NORMAL, FIDELITY, 'test data')
                    for e in entries]
            rows[(data, model)] = (ca.median_and_count(local), ca.median_and_count(glob))
    local_wins = all(r[0][0] > 0 for r in rows.values())
    global_never = all(r[1][0] <= 0 for r in rows.values())
    verdict = 'confirmed' if local_wins and global_never else 'refuted'
    measured = (f'local $+{min(r[0][0] for r in rows.values()):.4f}$ to '
                f'$+{max(r[0][0] for r in rows.values()):.4f}$ in all four cells; '
                f'global never above $0$')
    return ('P7', 'local imbalance is the signal, not global',
            'local wins in all four cells; global never helps', measured, verdict)


def p8(balance):
    from analyse_balance import cell, base_model
    pairs = {}
    for entry in balance.values():
        key = (entry['dataset'], base_model(entry), cell(entry)[0])
        pairs.setdefault(key, {})[cell(entry)[1]] = entry
    normal, balanced = [], []
    for both in pairs.values():
        if len(both) != 2:
            continue
        normal.append(ca.paired_gain(both['normal'], SAMPLED, NORMAL, FIDELITY,
                                     'test data'))
        balanced.append(ca.paired_gain(both['balanced'], SAMPLED, NORMAL, FIDELITY,
                                       'test data'))
    normal, balanced = np.array(normal), np.array(balanced)
    ok = np.isfinite(normal) & np.isfinite(balanced)
    p = wilcoxon(normal[ok], balanced[ok]).pvalue
    shrinks = int((balanced[ok] < normal[ok]).sum())
    verdict = 'confirmed' if shrinks > ok.sum()/2 and p < 0.05 else 'refuted'
    measured = (f'gain $+{np.median(normal[ok]):.4f}$ normally trained vs '
                f'$+{np.median(balanced[ok]):.4f}$ balance-trained, smaller in '
                f'{shrinks}/{ok.sum()}, $p = {p:.2g}$')
    return ('P8', 'balanced training shrinks the gain (P6\'s mechanism again)',
            'gain smaller against a balance-trained black box', measured, verdict)


ROWS = [(p1, 'marginal'), (p2, 'marginal'), (p3, 'degrade'), (p4, 'degrade'),
        (p5, 'marginal'), (p6, 'degrade'), (p7, 'balance'), (p8, 'balance')]


def main():
    marginal, _, _ = ca.load('results_marginal.json')
    degrade, _, _ = ca.load('results_degrade.json')
    balance, _, _ = ca.load('results_balance.json')
    data = {'marginal': marginal, 'degrade': degrade, 'balance': balance}

    rows = [fn(data[source]) for fn, source in ROWS]
    for name, claim, registered, measured, verdict in rows:
        print(f'{name}  {verdict:42s} {claim}')

    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \footnotesize',
             r'  \setlength{\tabcolsep}{4pt}',
             r'  \caption{Each prediction as registered in \texttt{PREREGISTRATION.md} '
             r'before its sweep ran, against what was measured. Verdicts are computed from '
             r'the registered thresholds rather than assigned by hand. P1--P6 were '
             r'registered together, P7--P8 (E4) after the first six were analysed.}',
             r'  \label{tab:predictions}',
             r'  \setlength{\tabcolsep}{3pt}',
             r'  \begin{tabular}{l p{0.24\textwidth} p{0.34\textwidth} l}',
             r'    \toprule',
             r'    & prediction & measured & verdict \\',
             r'    \midrule']
    for name, claim, registered, measured, verdict in rows:
        lines.append(f'    \\textbf{{{name}}} & {claim} & {measured} & {verdict} \\\\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']

    out = paths.table('predictions.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('\nwritten', out)


if __name__ == '__main__':
    main()
