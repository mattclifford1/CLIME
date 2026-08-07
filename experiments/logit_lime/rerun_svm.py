'''re-run only the SVM rows after the gamma fix and merge back into results.json'''
import json, numpy as np, warnings, clime
import sys; sys.path.insert(0, '.')
from experiment import opts, diagnostic, DATASETS, EXPLAINERS, METRICS
from clime.evaluation.key_points import get_points_between_class_means
warnings.filterwarnings('ignore')

d = json.load(open('results.json'))
for dataset in DATASETS:
    entry = {'metrics': {}}
    for metric in METRICS:
        entry['metrics'][metric] = {}
        for expl in EXPLAINERS:
            r = clime.pipeline.run_pipeline(opts(dataset, 'SVM', expl, metric), parallel_eval=False)
            s = np.array(r['score']['scores'])
            entry['metrics'][metric][expl] = {'mean': float(s.mean()), 'scores': s.tolist()}
            entry['model_stats'] = {k: float(v) for k, v in r['model_stats'].items()}
    base = clime.pipeline.run_pipeline(opts(dataset, 'SVM', EXPLAINERS[0], METRICS[0]), parallel_eval=False)
    qs, _ = get_points_between_class_means(base['test_data'])
    rl, rp, sat = diagnostic(base['clf'], base['train_data'], base['test_data'], qs)
    entry['diagnostic'] = {'r2_logit': rl, 'r2_prob': rp, 'gap': rl-rp, 'saturation': sat}
    entry['eval_points'] = np.array(qs).tolist()
    d[f'{dataset}|SVM'] = entry
    m = entry['metrics']['Brier score (local)']
    print(f"{dataset:26s} SVM  acc={entry['model_stats']['test accurracy']:.3f} "
          f"gap={rl-rp:+.3f} sat={sat:5.1%} "
          f"advantage={m['bLIMEy (normal)']['mean']/max(m['bLIMEy (logit)']['mean'],1e-30):8.2f}x", flush=True)
json.dump(d, open('results.json', 'w'))
print('merged')
