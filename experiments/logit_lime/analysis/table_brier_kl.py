'''
generate table1.tex

usage:  python gen_table.py [results_taxonomy.json]

Defaults to the taxonomy sweep rather than results.json: results.json predates the
per-query-point seeding fix, and the taxonomy sweep re-runs every configuration in this
table, so reading from it keeps the paper's numbers on one footing.
'''
import sys, json, numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_taxonomy.json'))
DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes']
MODELS = ['Logistic', 'MLP', 'SVM', 'Gradient Boosting', 'Random Forest',
          'Random Forest (Platt calibrated)', 'Random Forest (isotonic calibrated)']
NICE = {'Logistic': 'Logistic regression', 'MLP': 'MLP', 'SVM': 'SVM',
        'Gradient Boosting': 'Gradient boosting', 'Random Forest': 'Random forest',
        'Random Forest (Platt calibrated)': 'Random forest + Platt',
        'Random Forest (isotonic calibrated)': 'Random forest + isotonic'}
DS_SHORT = {'Gaussian': 'Gaussian', 'Breast Cancer': 'Breast Cancer',
            'Banknote Authentication': 'Banknote', 'Pima Indian Diabetes': 'Pima Diabetes'}
E = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']

def fmt(v, best):
    if v < 1e-4:
        m, e = f'{v:.1e}'.split('e')
        body = f'{m}\\!\\cdot\\!10^{{{int(e)}}}'
    else:
        body = f'{v:.4f}'.rstrip('0').rstrip('.')
    # N.B. \textbf does not bold maths - it has to be \mathbf inside the $...$
    return f'$\\mathbf{{{body}}}$' if best else f'${body}$' 

lines = [
 r'\begin{table}[t]', r'\centering', r'\footnotesize',
 r'\setlength{\tabcolsep}{4pt}',
 r'\caption{Local Brier score and local KL divergence between each surrogate and the black'
 r' box, averaged over the 20 query points. $\Delta$ is the log-odds linearity gap'
 r' (Eq.~\ref{eq:gap}), computed from the black box alone; \emph{sat.} is the fraction of'
 r' locally sampled points with a saturated probability. Best surrogate per row per metric'
 r' in bold. The large Logit-LIME gains occur exactly where $\Delta$ is large; calibrating'
 r' the random forest removes saturation without changing $\Delta$ or the outcome. Note that'
 r' the log.reg.\ variant often wins on Brier score while losing badly on KL'
 r' (Section~\ref{sec:hardlabel}).}',
 r'\label{tab:main}',
 r'\begin{tabular}{llrr rrr rrr}', r'\toprule',
 r'& & & & \multicolumn{3}{c}{local Brier score} & \multicolumn{3}{c}{local KL divergence} \\',
 r'\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
 r'Dataset & Black box & $\Delta$ & sat. & standard & logit & log.reg. & standard & logit & log.reg. \\',
 r'\midrule']

for di, ds in enumerate(DATASETS):
    for mi, model in enumerate(MODELS):
        e = d[f'{ds}|{model}']
        b = [e['metrics']['Brier score (local)'][x]['mean'] for x in E]
        k = [e['metrics']['KL divergence (local)'][x]['mean'] for x in E]
        bb, kb = int(np.argmin(b)), int(np.argmin(k))
        first = DS_SHORT[ds] if mi == 0 else ''
        cells = ' & '.join([fmt(v, i == bb) for i, v in enumerate(b)] +
                           [fmt(v, i == kb) for i, v in enumerate(k)])
        lines.append(f"{first} & {NICE[model]} & ${e['diagnostic']['gap']:+.2f}$ & "
                     f"${e['diagnostic']['saturation']*100:.0f}\\%$ & {cells} \\\\")
    if di < len(DATASETS)-1:
        lines.append(r'\addlinespace')

lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('table1.tex', 'w').write('\n'.join(lines) + '\n')
print('table1.tex written')

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
