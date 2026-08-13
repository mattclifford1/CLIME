'''
Kernel width sweep.

The locality kernel width k = scale * sqrt(n_features) defines "local" for both the
surrogate's training weights and the local evaluation metric. Because it sits on both
sides, a sceptic can reasonably ask whether the Logit-LIME effect is an artefact of
LIME's default scale of 0.75. This sweeps it.

Prediction: the *ordering* of surrogates is preserved across scales. At very small widths
every black box looks locally linear in both spaces, so the gap between surrogates should
shrink toward zero; at large widths the neighbourhood spans the whole sigmoid and the
advantage for linear-log-odds black boxes should grow.

usage:  python sweep_kernel.py <output.json>
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import sys
import os
import json
import warnings
import numpy as np
import clime
from clime.data.utils import costs
from sweeps.sweep import opts, diagnostic, EXPLAINERS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

SCALES = [0.15, 0.3, 0.5, 0.75, 1.25, 2.0, 3.0]     # 0.75 is the LIME default
DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes']
MODELS = ['Logistic', 'LDA', 'MLP', 'Random Forest', 'Gradient Boosting']
METRIC = 'Brier score (local)'


def run(out_path):
    out_path = paths.results(out_path)   # a bare name lands in results/

    out = {'_meta': {'scales': SCALES, 'default_scale': 0.75}}
    if os.path.exists(out_path):        # see the note on resuming in sweep.py
        done = json.load(open(out_path))
        out.update({k: v for k, v in done.items()
                    if not k.startswith('_') and 'error' not in v})
        print(f'resuming: {len(out)-1} configurations already done', flush=True)

    for scale in SCALES:
        # set on the module so both the surrogate weights and the metric use it
        costs.KERNEL_WIDTH_SCALE = scale
        # the pipeline caches on opts, which does not include the kernel width
        clime.pipeline.run_pipeline.cache_clear()
        for dataset in DATASETS:
            for model in MODELS:
                if f'{scale}|{dataset}|{model}' in out:
                    continue
                entry = {}
                try:
                    for expl in EXPLAINERS:
                        r = clime.pipeline.run_pipeline(opts(dataset, model, expl, METRIC),
                                                        parallel_eval=False)
                        s = np.array(r['score']['scores'])
                        entry[expl] = {'mean': float(s.mean()), 'scores': s.tolist()}
                    base = clime.pipeline.run_pipeline(opts(dataset, model, EXPLAINERS[0], METRIC),
                                                       parallel_eval=False)
                    qs, _ = get_points_between_class_means(base['test_data'])
                    rl, rp, sat = diagnostic(base['clf'], base['test_data'], qs)
                    entry['diagnostic'] = {'r2_logit': rl, 'r2_prob': rp, 'gap': rl-rp,
                                           'saturation': sat}
                except Exception as e:
                    entry['error'] = f'{type(e).__name__}: {e}'
                out[f'{scale}|{dataset}|{model}'] = entry
                if 'error' not in entry:
                    adv = entry[EXPLAINERS[0]]['mean']/max(entry[EXPLAINERS[1]]['mean'], 1e-30)
                    print(f"scale={scale:<5} {dataset:26s} {model:22s} "
                          f"gap={entry['diagnostic']['gap']:+.3f} advantage={adv:10.2f}x",
                          flush=True)
                else:
                    print(f"scale={scale:<5} {dataset:26s} {model:22s} FAILED", flush=True)
                json.dump(out, open(out_path, 'w'))
    costs.KERNEL_WIDTH_SCALE = 0.75
    json.dump(out, open(out_path, 'w'))
    print('written', out_path)


if __name__ == '__main__':
    run(sys.argv[1])
