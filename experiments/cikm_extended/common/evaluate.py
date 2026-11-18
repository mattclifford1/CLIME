'''one job: a dataset split and a model family, every arm explained at every query point'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import get_local_points, get_points_between_class_means

from . import boxes, scores, surrogates, tasks

LINE_POINTS = 20
TEST_POINTS_PER_CLASS = 10
LOCAL_EVAL_SAMPLES = 500


def query_points(task, line=LINE_POINTS, per_class=TEST_POINTS_PER_CLASS):
    '''the CIKM'23 line through the test class means, and random test points of each class'''
    test = task.test
    out = []
    if line:
        points, _ = get_points_between_class_means(test, num_samples=line)
        out += [('line', i, np.nan, np.asarray(q, dtype=np.float64)) for i, q in enumerate(points)]
    rng = np.random.default_rng([task.seed, 20261117])
    for label in (0, 1):
        pool = np.flatnonzero(test['y'] == label)
        for i in rng.choice(pool, size=min(per_class, len(pool)), replace=False):
            out.append(('test', int(i), label, np.asarray(test['X'][i], dtype=np.float64)))
    return out


def _evaluation_sets(box, task, query_point):
    '''(black-box P(1), black-box class, kernel weight, X) on the test set and on a local sample'''
    out = {}
    local = get_local_points(task.test, query_point, samples=LOCAL_EVAL_SAMPLES)['X']
    for name, X in (('test', task.test['X']), ('local', local)):
        prob = box.predict_proba(X)
        out[name] = (prob[:, 1], np.argmax(prob, axis=1).astype(np.int64),
                     costs.weights_based_on_distance(query_point, X), X)
    return out


def explain(box, task, query_point, deltas=surrogates.FAST_DELTAS, matched=False):
    '''a row of neighbourhood diagnostics and one row of scores per explainer'''
    built, lime, balanced = surrogates.build(box, query_point, task.test, deltas, matched)
    sample = lime.sampled
    label = np.asarray(sample['y']).astype(np.int64)
    weight = costs.weights_based_on_distance(query_point, sample['X'])
    sets = _evaluation_sets(box, task, query_point)

    share = float(label.mean())
    f_query = float(box.predict_proba([query_point])[0, 1])
    query = {'f_query': f_query, 'f_class': int(f_query > 0.5),
             'sample_share_1': share, 'sample_minority': min(share, 1 - share),
             'sample_weighted_share_1': float((weight*label).sum()/weight.sum()),
             'test_weighted_share_1': float((sets['test'][2]*sets['test'][1]).sum()
                                            / sets['test'][2].sum()),
             'test_effective_share': scores.effective_share(sets['test'][2]),
             **scores.explanation_agreement(lime.get_explanation(), balanced.get_explanation())}

    rows = []
    shared = {}
    for name, (surrogate, cut) in built.items():
        row = {'explainer': name, 'cut': cut}
        for where, (f_prob, f_class, w, X) in sets.items():
            key = (id(surrogate), where)
            if key not in shared:
                g = surrogate.score(X)
                shared[key] = (g, scores.ranking(g, f_class, w), scores.brier(g, f_prob, w))
            g, auc, brier = shared[key]
            row.update({f'{where}_{k}': v
                        for k, v in scores.operating_point(g, cut, f_class, w).items()})
            row[f'{where}_auc'], row[f'{where}_brier'] = auc, brier
        rows.append(row)
    return query, rows


def described(task, tags, model_rows):
    '''the model rows of a job, tagged with the job and the size of its data'''
    return [{**tags, 'ratio': task.ratio, 'n_features': task.train['X'].shape[1],
             'n_train': len(task.train['y']), 'n_test': len(task.test['y']), **r}
            for r in model_rows]


def run(dataset, setting, seed, family, arms=boxes.TRAINING_ARMS, deltas=None,
        surrogate_deltas=surrogates.FAST_DELTAS, line=LINE_POINTS,
        per_class=TEST_POINTS_PER_CLASS, kernel_width=None, matched=False):
    '''
    {'models', 'queries', 'scores', 'failures'}: lists of rows for one (dataset, setting,
    seed, family); a query point that raises goes to 'failures' and is skipped

    deltas: methods placed on the black box (all by default, () for none).
    surrogate_deltas: methods placed on LIME's surrogate, for the training arms and the
    re-thresholded black box; a deltas-placed black box gets LIME, CLIME and the
    re-thresholded surrogate only. matched: add LIME cut at CLIME's operating point.
    '''
    clime.RANDOM_SEED = int(seed)
    costs.KERNEL_WIDTH_SCALE = 0.75 if kernel_width is None else kernel_width
    task = tasks.load(dataset, setting, seed)
    built, model_rows = boxes.build(task, family, arms, deltas)
    tags = {'dataset': dataset, 'setting': setting, 'seed': seed, 'family': family,
            'kernel_width': costs.KERNEL_WIDTH_SCALE}
    out = {'models': described(task, tags, model_rows), 'queries': [], 'scores': []}

    points = query_points(task, line, per_class)
    placement = {r['arm']: r['placement'] for r in model_rows}
    cut = {r['arm']: r['threshold'] for r in model_rows}
    done, failures = {}, []
    for arm, box in built.items():
        on_surrogate = () if placement[arm] == 'deltas' else surrogate_deltas
        # a placed threshold equal to an earlier one explains exactly as that arm does
        twin = None
        if placement[arm] == 'deltas':
            twin = next((a for a in done if placement[a] != 'validation'
                         and (placement[a] == 'deltas' or a == 'plain')
                         and abs(cut[a] - cut[arm]) < 1e-9), None)
        done[arm] = {}
        for kind, index, label, q in points:
            where = {**tags, 'arm': arm, 'kind': kind, 'index': index, 'label': label}
            if twin is not None:
                if (kind, index) not in done[twin]:
                    continue
                query, rows = done[twin][(kind, index)]
                rows = [r for r in rows if not r['explainer'].startswith(surrogates.PREFIX)
                        or r['explainer'] == surrogates.RETHRESHOLDED]
            else:
                try:
                    query, rows = explain(box, task, q, on_surrogate, matched)
                except Exception as e:   # noqa: BLE001 - one bad query point, not the job
                    failures.append({**where, 'error': f'{type(e).__name__}: {e}'})
                    continue
            done[arm][(kind, index)] = (query, rows)
            out['queries'].append({**where, **query})
            out['scores'] += [{**where, **r} for r in rows]
    out['failures'] = failures
    return out
