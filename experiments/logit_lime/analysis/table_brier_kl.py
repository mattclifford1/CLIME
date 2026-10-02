'''
generate tables/brier-kl-by-blackbox.tex (Table 1 of the paper)

usage:  python analysis/table_brier_kl.py [results_taxonomy.json]

Defaults to the taxonomy sweep rather than results.json: results.json predates the
per-query-point seeding fix, and the taxonomy sweep re-runs every configuration in this
table, so reading from it keeps the paper's numbers on one footing.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys, json, numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else paths.results('results_taxonomy.json')))
# the soft-label logistic surrogate (sixth registration), scored on the same evaluation points
soft = json.load(open(paths.results('results_soft_logistic_full.json')))
DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes']
MODELS = ['Logistic', 'MLP', 'SVM', 'Gradient Boosting', 'Random Forest',
          'Random Forest (Platt calibrated)', 'Random Forest (isotonic calibrated)']
# short labels: with four surrogates per metric the table only fits the text width with them
NICE = {'Logistic': 'Logistic', 'MLP': 'MLP', 'SVM': 'SVM',
        'Gradient Boosting': 'Grad.\\ boosting', 'Random Forest': 'Random forest',
        'Random Forest (Platt calibrated)': 'RF + Platt',
        'Random Forest (isotonic calibrated)': 'RF + isotonic'}
DS_SHORT = {'Gaussian': 'Gaussian', 'Breast Cancer': 'Breast Cancer',
            'Banknote Authentication': 'Banknote', 'Pima Indian Diabetes': 'Pima Diabetes'}
E = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']

def body(v):
    if v < 1e-4:
        m, e = f'{v:.1e}'.split('e')
        return f'{m}\\!\\cdot\\!10^{{{int(e)}}}'
    return f'{v:.4f}'.rstrip('0').rstrip('.')


def cells(values):
    '''bold every value that ties the best at the precision it is printed to'''
    shown = [body(v) for v in values]
    best = shown[int(np.argmin(values))]
    # N.B. \textbf does not bold maths - it has to be \mathbf inside the $...$
    return [f'$\\mathbf{{{b}}}$' if b == best else f'${b}$' for b in shown]

lines = [
 r'\begin{table}[t]', r'\centering', r'\scriptsize',
 r'\setlength{\tabcolsep}{2.5pt}',
 r'\caption{Local Brier score and local KL divergence between each surrogate and the black'
 r' box, averaged over the 20 query points. $\Rlogit$ is the diagnostic of'
 r' Eq.~\ref{eq:r2}, the in-sample fit of an unregularised Logit-LIME; \emph{sat.} is the'
 r' fraction of locally sampled points with a saturated probability. Best surrogate per row'
 r' per metric in bold. The large Logit-LIME gains occur exactly where $\Rlogit$ is close to'
 r' $1$; calibrating the random forest does not produce one, because $\Rlogit$ stays far'
 r' from $1$. Platt scaling removes saturation and isotonic calibration does not, and which of'
 r' standard LIME and Logit-LIME wins can change under calibration'
 r' (Section~\ref{sec:saturation}). \emph{hard} and \emph{soft} are both logistic regressions:'
 r' the first fitted to the black box\textquoteright s rounded classes, which often beats'
 r' standard LIME and Logit-LIME on Brier score while losing badly on KL'
 r' (Section~\ref{sec:hardlabel}); the second to its probabilities, which is best in most'
 r' rows on both metrics (Section~\ref{sec:softlabel}). Ties at the printed precision are'
 r' all bold.}',
 r'\label{tab:main}',
 r'\begin{tabular}{@{}llrr rrrr rrrr@{}}', r'\toprule',
 r'& & & & \multicolumn{4}{c}{local Brier score} & \multicolumn{4}{c}{local KL divergence} \\',
 r'\cmidrule(lr){5-8}\cmidrule(lr){9-12}',
 r'Dataset & Black box & $\Rlogit$ & sat. & standard & logit & hard & soft & standard & logit & hard & soft \\',
 r'\midrule']

for di, ds in enumerate(DATASETS):
    for mi, model in enumerate(MODELS):
        e = d[f'{ds}|{model}']
        b = [e['metrics']['Brier score (local)'][x]['mean'] for x in E]
        k = [e['metrics']['KL divergence (local)'][x]['mean'] for x in E]
        s_ = soft[f'{ds}|{model}']
        # the soft-label numbers come from another file: refuse to put them beside a
        # results file from a different seed or split, whose neighbourhoods differ
        q0 = e['metrics']['Brier score (local)'][E[0]]['scores'][0]
        assert abs(q0 - s_['check_standard_brier_q0']) <= 1e-9*max(abs(q0), 1e-12), \
            f'{ds}|{model}: results_soft_logistic_full.json does not join to this file'
        b.append(s_['mean Brier | local sample'])
        k.append(s_['mean KL | local sample'])
        first = DS_SHORT[ds] if mi == 0 else ''
        row = ' & '.join(cells(b) + cells(k))
        lines.append(f"{first} & {NICE[model]} & ${e['diagnostic']['r2_logit']:.3f}$ & "
                     f"${e['diagnostic']['saturation']*100:.0f}\\%$ & {row} \\\\")
    if di < len(DATASETS)-1:
        lines.append(r'\addlinespace')

lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(paths.table('brier-kl-by-blackbox.tex'), 'w').write('\n'.join(lines) + '\n')
print('tables/brier-kl-by-blackbox.tex written')

# summary numbers quoted in the text, so the prose cannot drift from the data
print('\n--- numbers used in the prose ---')
for ds, model in [('Gaussian', 'Logistic'), ('Breast Cancer', 'Logistic')]:
    m = d[f'{ds}|{model}']['metrics']['Brier score (local)']
    print(f"{ds}/{model}: standard={m[E[0]]['mean']:.2e} logit={m[E[1]]['mean']:.2e} "
          f"ratio={m[E[0]]['mean']/m[E[1]]['mean']:.3g}")
for model in ['Random Forest', 'Random Forest (Platt calibrated)']:
    e = d[f'Gaussian|{model}']
    m = e['metrics']['Brier score (local)']
    print(f"Gaussian/{model}: sat={e['diagnostic']['saturation']:.1%} "
          f"gap={e['diagnostic']['gap']:+.3f} ratio={m[E[0]]['mean']/m[E[1]]['mean']:.3g}")
from scipy.stats import spearmanr
# only the configurations shown in this table - the taxonomy file holds many more, and
# the correlation over the full set is reported separately by analyse.py
rows = [(v['diagnostic']['gap'], v['diagnostic']['saturation'],
         v['metrics']['Brier score (local)'][E[0]]['mean']/max(v['metrics']['Brier score (local)'][E[1]]['mean'],1e-12))
        for v in (d[f'{ds}|{m}'] for ds in DATASETS for m in MODELS)]
g, s, a = zip(*rows)
print(f"n={len(rows)}  spearman(gap, advantage)={spearmanr(g,a)[0]:.3f} p={spearmanr(g,a)[1]:.2g}")
print(f"          spearman(sat, advantage)={spearmanr(s,a)[0]:.3f} p={spearmanr(s,a)[1]:.2g}")
