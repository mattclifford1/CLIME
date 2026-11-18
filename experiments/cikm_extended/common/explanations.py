'''one job: six surrogates at every query point, scored as predictions and as explanations'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from clime.data.utils import costs

from . import boxes, evaluate, gradients, scores, surrogates, tasks

EXPLAINERS = {
    'LIME': {},
    'CLIME': {'class_weight_sampled': True},
    'Logit-LIME': {'train_logits': True},
    'Logit-CLIME': {'train_logits': True, 'class_weight_sampled': True},
    'Soft logistic': {'soft_logistic_regression': True},
    'Soft logistic CLIME': {'soft_logistic_regression': True, 'class_weight_sampled': True},
}
# LIME fitted to the plain black box, cut at the moved threshold
THRESHOLD_ONLY = 'LIME, threshold only'
SWEPT = ('LIME', 'CLIME')
CUTS = (0.01, 0.02, 0.05, *np.round(np.arange(0.1, 0.91, 0.05), 2), 0.95, 0.98, 0.99)
DELTAS = ('CP Minimax',)
REDRAW = 'surrogate training sample, second draw'
GRADIENT = 'gradient'


def _fit(box, q, test, **extra):
    return {name: surrogates.kept(box, q, test_data=test, **flags, **extra)
            for name, flags in EXPLAINERS.items()}


def _truths(box, q, local):
    '''
    the black box's log-odds gradient at the query point, and its kernel-weighted mean over
    the local sample in log-odds and in probability; empty for a family with no gradient
    '''
    at_query = gradients.log_odds(box, q[None])
    if at_query is None:
        return {}
    f_prob, _, w, X = local
    G = gradients.log_odds(box, X)
    return {'point': at_query[0], 'mean': np.average(G, axis=0, weights=w),
            'mean_probability': np.average(G*(f_prob*(1 - f_prob))[:, None], axis=0, weights=w)}


def _agreement(prefix, a, b, keep=('cosine', 'same_top', 'rank', 'norm_ratio')):
    found = scores.explanation_agreement(a, b)
    return {f'{prefix}_{k}': found[k] for k in keep}


def _scored(where, g, cut, f_prob, f_class, w):
    out = {**scores.operating_point(g, cut, f_class, w), 'auc': scores.ranking(g, f_class, w),
           **scores.calibration(g, f_prob, f_class, w)}
    return {f'{where}_{k}': v for k, v in out.items()}


def explain(box, task, q, plain_lime=None, sweep=False):
    '''
    (query row, score rows, cut rows, {explainer: coefficients}, gradients, LIME's model)

    plain_lime: LIME's model for the plain black box, scored at this box's moved threshold.
    sweep: also score LIME and CLIME at every cut of CUTS.
    '''
    fitted = _fit(box, q, task.test)
    again = _fit(box, q, task.test, sample_salt=REDRAW)
    sample = fitted['LIME'].sampled
    f_sample = sample['p(y|x)'][:, 1]
    share = float(np.asarray(sample['y']).mean())
    sets = evaluate._evaluation_sets(box, task, q)
    truths = _truths(box, q, sets['local'])
    # the best constant surrogate under LIME's weights and under CLIME's
    constant = {'': np.average(f_sample, weights=fitted['LIME']._get_sampled_weights(sample)),
                '_balanced': np.average(f_sample,
                                        weights=fitted['CLIME']._get_sampled_weights(sample))}

    f_query = float(box.predict_proba([q])[0, 1])
    query = {'f_query': f_query, 'f_class': int(f_query > 0.5),
             'sample_minority': min(share, 1 - share), 'has_gradient': bool(truths)}
    for where, (f_prob, f_class, w, X) in sets.items():
        for suffix, c in constant.items():
            flat = scores.calibration(np.full(len(X), c), f_prob, f_class, w)
            for loss in ('brier', 'kl'):
                query[f'{where}_{loss}{suffix}_constant'] = flat[f'{loss}{suffix}']

    rows, cut_rows, coefs = [], [], {}
    for name, surrogate in fitted.items():
        coefs[name] = coef = surrogate.get_explanation()
        g = {where: surrogate.score(s[3]) for where, s in sets.items()}
        row = {'explainer': name, 'cut': 0.5, 'coef_norm': float(np.linalg.norm(coef)),
               **_agreement('stability', coef, again[name].get_explanation())}
        for truth, vector in truths.items():
            row.update(_agreement(f'truth_{truth}', vector, coef, ('cosine', 'same_top', 'rank')))
        for where, (f_prob, f_class, w, _) in sets.items():
            row.update(_scored(where, g[where], 0.5, f_prob, f_class, w))
        rows.append(row)
        if sweep and name in SWEPT:
            for cut in CUTS:
                at = {'explainer': name, 'cut': cut}
                for where, (_, f_class, w, _) in sets.items():
                    found = scores.operating_point(g[where], cut, f_class, w)
                    at.update({f'{where}_{k}': v for k, v in found.items()})
                cut_rows.append(at)
    if plain_lime is not None:
        row = {'explainer': THRESHOLD_ONLY, 'cut': box.probability}
        for where, (_, f_class, w, X) in sets.items():
            g = plain_lime.predict(X)[:, 1]
            found = {**scores.operating_point(g, box.probability, f_class, w),
                     'auc': scores.ranking(g, f_class, w)}
            row.update({f'{where}_{k}': v for k, v in found.items()})
        rows.append(row)
    return query, rows, cut_rows, coefs, truths, fitted['LIME'].surrogate_model


def run(dataset, setting, seed, family, arms=boxes.TRAINING_ARMS, deltas=DELTAS):
    '''
    {'models', 'queries', 'scores', 'cuts', 'pairs', 'failures'}: lists of rows for one
    (dataset, setting, seed, family); 'pairs' compares each arm's explanations, and the
    black box's gradient, with the plain arm's at the same query point
    '''
    clime.RANDOM_SEED = int(seed)
    costs.KERNEL_WIDTH_SCALE = 0.75
    task = tasks.load(dataset, setting, seed)
    built, model_rows = boxes.build(task, family, arms, deltas)
    tags = {'dataset': dataset, 'setting': setting, 'seed': seed, 'family': family}
    out = {'models': evaluate.described(task, tags, model_rows),
           'queries': [], 'scores': [], 'cuts': [], 'pairs': [], 'failures': []}
    placement = {r['arm']: r['placement'] for r in model_rows}
    plain = {}
    for arm, box in built.items():
        moved = placement[arm] != 'training'
        for kind, index, label, q in evaluate.query_points(task):
            where = {**tags, 'arm': arm, 'kind': kind, 'index': index, 'label': label}
            first = plain.get((kind, index))
            try:
                query, rows, cut_rows, coefs, truths, lime = explain(
                    box, task, q, plain_lime=first[2] if moved and first else None,
                    sweep=arm == 'plain')
            except Exception as e:   # noqa: BLE001 - one bad query point, not the job
                out['failures'].append({**where, 'error': f'{type(e).__name__}: {e}'})
                continue
            out['queries'].append({**where, **query})
            out['scores'] += [{**where, **r} for r in rows]
            out['cuts'] += [{**where, **r} for r in cut_rows]
            if arm == 'plain':
                plain[(kind, index)] = (coefs, truths, lime)
            elif first:
                pairs = {name: (first[0][name], coef) for name, coef in coefs.items()}
                if truths:
                    pairs[GRADIENT] = (first[1]['point'], truths['point'])
                out['pairs'] += [{**where, 'explainer': name, **scores.explanation_agreement(a, b)}
                                 for name, (a, b) in pairs.items()]
    return out
