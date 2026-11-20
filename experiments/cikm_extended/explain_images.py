'''
Six surrogates and KernelSHAP fitted to patch masks of every query image, from the logits
stored by run_images.py, scored as explanations. Writes results/raw/images_explained/.

usage: N_JOBS=8 uv run python explain_images.py
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from clime.models import threshold
from common import patches, paths, runner, scores

SAMPLES = (100, 300, 1000, 3000, 10_000)
DEFAULT = 1000   # lime_image's number of masks
DELTAS = 'CP Minimax'


def placements(data):
    '''{arm: shift of the logit}: the network as trained, and its threshold moved where it is plain'''
    arm, ratio = str(data['arm']), int(data['ratio'])
    out = {arm: 0.0}
    if arm == 'plain' and ratio > 1:
        out['prior shift'] = float(np.log(ratio))
        out['re-thresholded'] = -threshold.balanced_accuracy_threshold(data['val'], data['y_val'])
        out[DELTAS] = -threshold.deltas_threshold(data['train'], data['y_train'], DELTAS)[0]
    return out


def described(z, y, shift):
    pred = z + shift > 0
    r0, r1 = float((~pred[y == 0]).mean()), float(pred[y == 1].mean())
    return {'shift': shift, 'balanced_accuracy': (r0 + r1)/2, 'recall_majority': r0,
            'recall_minority': r1, 'positive_rate': float(pred.mean()),
            'roc_auc': float(roc_auc_score(y, z))}


def draws(seed, query):
    '''two draws of mask indices, the first row the full image as lime_image has it'''
    out = []
    for salt in (0, 1):
        rng = np.random.default_rng([int(seed), int(query), salt])
        index = rng.integers(0, patches.FULL + 1, max(SAMPLES))
        index[0] = patches.FULL
        out.append(index)
    return out


def shap_seed(seed, image, salt, n):
    return int(np.random.SeedSequence([int(seed), int(image), salt, n]).generate_state(1)[0])


def explain(z, first, second, M, w, seed, image):
    '''(query row, score rows, sample rows, coefficient rows) for one image's logits on every mask'''
    f_prob, f_class = expit(z), (z > 0).astype(np.int64)
    truths = {'leave one out': patches.leave_one_out(z), 'Shapley': patches.shapley(z)}
    share, weighted = f_class.mean(), np.average(f_class, weights=w)
    sample = first[:DEFAULT]
    query = {'logit': float(z[patches.FULL]), 'f_class': int(z[patches.FULL] > 0),
             'minority': min(share, 1 - share), 'minority_weighted': min(weighted, 1 - weighted),
             'sample_minority': min(f_class[sample].mean(), 1 - f_class[sample].mean())}
    for suffix, weights in (('', w[sample]),
                            ('_balanced', w[sample]*patches.class_weights(f_class[sample]))):
        flat = np.full(len(z), np.average(f_prob[sample], weights=weights))
        query[f'brier{suffix}_constant'] = scores.calibration(flat, f_prob, f_class, w)[f'brier{suffix}']
    rows, sample_rows = [], []
    coefs = [{'explainer': name, 'fit': 'truth', 'coef': vector} for name, vector in truths.items()]
    for name in (*patches.EXPLAINERS, *patches.SHAP):
        if name in patches.SHAP:
            exact, _ = patches.exact_shap(name, z)
            fitted = {n: patches.fit_shap(name, z, n, shap_seed(seed, image, 2, n)) for n in SAMPLES}
            again, _ = patches.fit_shap(name, z, DEFAULT, shap_seed(seed, image, 3, DEFAULT))
        else:
            exact, _ = patches.fit(name, M, z, w)
            fitted = {n: patches.fit(name, M[first[:n]], z[first[:n]], w[first[:n]]) for n in SAMPLES}
            again, _ = patches.fit(name, M[second[:DEFAULT]], z[second[:DEFAULT]], w[second[:DEFAULT]])
        coef, intercept = fitted[DEFAULT]
        g = patches.predict(name, coef, intercept, M)
        row = {'explainer': name, 'coef_norm': float(np.linalg.norm(coef)),
               **scores.operating_point(g, 0.5, f_class, w), 'auc': scores.ranking(g, f_class, w),
               **scores.calibration(g, f_prob, f_class, w)}
        compared = {'stability': (coef, again), 'converged': (exact, coef)}
        for truth, vector in truths.items():
            key = truth.replace(' ', '_').lower()
            compared[f'truth_{key}'] = (vector, coef)
            compared[f'exact_{key}'] = (vector, exact)
        for key, (a, b) in compared.items():
            found = scores.explanation_agreement(a, b)
            row.update({f'{key}_cosine': found['cosine'], f'{key}_same_top': found['same_top']})
        rows.append(row)
        sample_rows += [{'explainer': name, 'samples': n,
                         **scores.explanation_agreement(exact, fitted[n][0])} for n in SAMPLES]
        coefs += [{'explainer': name, 'fit': 'sample', 'coef': coef},
                  {'explainer': name, 'fit': 'exact', 'coef': exact}]
    return query, rows, sample_rows, coefs


def run(path):
    data = np.load(path)
    tags = {k: data[k].item() for k in ('network', 'ratio', 'dataset', 'seed')}
    tags['trained'] = str(data['arm'])
    M = patches.masks()
    w = patches.kernel(M)
    out = {'models': [], 'queries': [], 'samples': [], 'coefs': [], 'scores': []}
    for arm, shift in placements(data).items():
        out['models'].append({**tags, 'arm': arm, **described(data['test'], data['y_test'], shift)})
        for i, query in enumerate(data['query']):
            where = {**tags, 'arm': arm, 'index': int(query), 'label': int(data['y_query'][i])}
            z = data['masks'][i].astype(np.float64) + shift
            found, rows, sample_rows, coefs = explain(z, *draws(tags['seed'], query), M, w,
                                                      tags['seed'], query)
            out['queries'].append({**where, **found})
            out['scores'] += [{**where, **r} for r in rows]
            out['samples'] += [{**where, **r} for r in sample_rows]
            out['coefs'] += [{**where, **r} for r in coefs]
    return out


if __name__ == '__main__':
    files = sorted(paths.raw('images').glob('*.npz'))
    runner.run_jobs('images_explained', [((f.stem,), {'path': f}) for f in files], job=run)
