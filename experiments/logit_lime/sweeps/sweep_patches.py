'''
Does the choice of space still matter in the interpretable domain?

The paper isolates its effect by fitting every surrogate in the raw feature space, and
lists the interpretable-domain transform as future work: LIME on images does not show the
surrogate pixels, it shows it binary "patch present / patch absent" indicators, and that
transform is known to introduce biases of its own. This sweep answers the question it
leaves open.

The transform is in common/patches.py, along with the reason the question is answerable:
for a black box with linear log-odds the truth survives into the patch domain in closed
form, gamma_j = sum_{i in patch j} beta_i (q_i - b_i), so each surrogate's patch weights
can be scored against a ground truth exactly as in sweep_ground_truth.py. The identity is
checked numerically per configuration rather than assumed - `linearity_residual` below.

Scope, stated plainly: one dataset, because Digits 3 vs 8 is the only registered dataset
whose features have a spatial layout to cut into patches, and the three exactly-linear
black boxes, because they are the ones with a ground truth. This is a controlled
demonstration that the effect survives the transform, not a grid.

Unlike the rest of the study the query points are real test images rather than points on
the line between the class means: a patch explanation is a statement about an actual
image, and interpolated points are not images. Twenty of them, class balanced.

Fidelity is measured in the interpretable domain too - on a fresh sample of z with its own
seeding salt, so no surrogate is scored on its own training sample. The metrics are
written out here rather than taken from clime.evaluation because those call the black box
on the points they are given, and these points are 16-dimensional patch vectors that the
64-dimensional black box cannot consume.

usage:  python sweeps/sweep_patches.py [results_patches.json]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, patches as P
from common import gradients

import json
import warnings
import numpy as np
from scipy.stats import spearmanr
import clime
from sweeps.sweep import opts, METRICS

warnings.filterwarnings('ignore')

DATASET = 'Digits 3 vs 8'
MODELS = ['Logistic', 'LDA', 'Nearest Class Mean']    # the exactly-linear families
N_POINTS = 20
EPS = 1e-12


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


def brier(g, f, w):
    return float(np.sum(w*(g - f)**2)/np.sum(w))


def kl(g, f, w):
    f = np.clip(f, EPS, 1 - EPS)
    g = np.clip(g, EPS, 1 - EPS)
    d = f*np.log(f/g) + (1 - f)*np.log((1 - f)/(1 - g))
    return float(max(np.sum(w*d)/np.sum(w), 0.0))


def query_images(test, n=N_POINTS):
    '''n real test images, class balanced, deterministic in the test split's own order'''
    y = np.asarray(test['y'])
    per = n//2
    idx = np.concatenate([np.where(y == 0)[0][:per], np.where(y == 1)[0][:per]])
    return test['X'][idx].astype(float), y[idx]


def run(out_path):
    out_path = paths.results(out_path)
    out = {}
    if os.path.exists(out_path):
        out = {k: v for k, v in json.load(open(out_path)).items() if 'error' not in v}
        print(f'resuming: {len(out)} done', flush=True)

    grid = P.patch_indices()
    for model in MODELS:
        key = f'{DATASET}|{model}'
        if key in out:
            continue
        entry = {}
        try:
            r = clime.pipeline.run_pipeline(
                opts(DATASET, model, 'bLIMEy (normal)', METRICS[0]), parallel_eval=False)
            clf, test = r['clf'], r['test_data']
            Q, _ = query_images(test)

            acc = {s: {'rho': [], 'top1': [], 'cos': [], 'brier': [], 'kl': []}
                   for s in ('standard', 'logit')}
            residuals, sat_fracs, wins = [], [], 0

            for q in Q:
                beta = np.asarray(gradients.grad_logit(clf, model, q[None, :])[0], float)
                baseline = np.zeros_like(q)
                truth = P.true_patch_coefficients(beta, q, baseline, grid)
                residual, _ = P.validate_linearity(clf, truth, q, baseline, grid)
                residuals.append(residual)

                # fidelity is scored on a fresh draw of z, never the training one
                Ze = P.sample_Z(len(grid), q, salt=P.EVAL_SALT)
                we = P.kernel_weights(Ze)
                fe = clf.predict_proba(P.compose(Ze, q, baseline, grid))[:, 1]
                # what the surrogates are actually fitted on: a probability pinned to 0 or
                # 1 in float64 carries no log-odds information, however extreme the truth
                sat_fracs.append(float(np.mean((fe <= P.EPS) | (fe >= 1 - P.EPS))))

                per = {}
                for label, as_logit in (('standard', False), ('logit', True)):
                    e = P.PatchLIME(clf, q, grid, baseline=baseline, train_logits=as_logit)
                    c = np.asarray(e.get_explanation(), dtype=float)
                    rho = spearmanr(np.abs(c), np.abs(truth))[0]
                    acc[label]['rho'].append(rho if np.isfinite(rho) else np.nan)
                    acc[label]['top1'].append(
                        float(np.argmax(np.abs(c)) == np.argmax(np.abs(truth))))
                    per[label] = cosine(c, truth)
                    acc[label]['cos'].append(per[label])
                    ge = e.predict_proba(Ze)[:, 1]
                    acc[label]['brier'].append(brier(ge, fe, we))
                    acc[label]['kl'].append(kl(ge, fe, we))
                wins += int(per['logit'] > per['standard'])

            for label in acc:
                for stat in acc[label]:
                    entry[f'{stat}_{label}'] = float(np.nanmean(acc[label][stat]))
            entry['n_points'] = int(len(Q))
            entry['n_patches'] = int(len(grid))
            entry['logit_better_cos'] = int(wins)
            entry['linearity_residual'] = float(np.nanmax(residuals))
            entry['saturated_fraction'] = float(np.mean(sat_fracs))
            print(f"{model:22s} cos std={entry['cos_standard']:.3f} "
                  f"logit={entry['cos_logit']:.3f}   top1 {entry['top1_standard']:.2f}/"
                  f"{entry['top1_logit']:.2f}   logit better {wins}/{len(Q)}   "
                  f"KL {entry['kl_standard']:.2e}/{entry['kl_logit']:.2e}   "
                  f"residual {entry['linearity_residual']:.1e}   "
                  f"saturated {entry['saturated_fraction']:.1%}", flush=True)
        except Exception as e:
            entry['error'] = f'{type(e).__name__}: {e}'
            print(f'{key:40s} FAILED {entry["error"][:70]}', flush=True)
        out[key] = entry
        json.dump(out, open(out_path, 'w'), indent=1)

    json.dump(out, open(out_path, 'w'), indent=1)
    print('written', out_path)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'results_patches.json')
