'''
Main experiment: when does a logit-space surrogate beat a probability-space one?

For every (dataset, black box) pair we measure
  (a) the surrogate quality of each explainer along the between-class-means line, and
  (b) a diagnostic computed from the BLACK BOX ALONE: how much better a weighted linear
      model fits its log-odds than its probabilities, on exactly the points the surrogate
      would be trained on.

The calibrated random forests are the discriminating case: Platt scaling removes
saturation but composes a sigmoid with the same piecewise-constant score, so it should
NOT create logit-linearity. If saturation were the mechanism, calibration would fix it.
'''
import sys, json, warnings
import numpy as np
import clime
from clime.data.utils import costs
from clime.evaluation.key_points import get_points_between_class_means, get_local_points

warnings.filterwarnings('ignore')

DATASETS = ['Gaussian', 'Breast Cancer', 'Banknote Authentication', 'Pima Indian Diabetes']
MODELS = ['Logistic', 'MLP', 'SVM', 'Gradient Boosting', 'Random Forest',
          'Random Forest (Platt calibrated)', 'Random Forest (isotonic calibrated)']
EXPLAINERS = ['bLIMEy (normal)', 'bLIMEy (logit)', 'bLIMEy (logistic regression)']
METRICS = ['Brier score (local)', 'KL divergence (local)']

DATA_PARAMS = {'class_samples': [200, 200], 'percent_of_data': 0.1, 'moons_noise': 0.2,
               'gaussian_means': [[-1, -1], [1, 1]],
               'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]}


def opts(dataset, model, explainer, metric):
    return {'dataset': dataset, 'data params': DATA_PARAMS, 'standardise data': True,
            'dataset rebalancing': 'none', 'model': model, 'model balancer': 'none',
            'explainer': explainer, 'evaluation metric': metric,
            'evaluation points': 'between_class_means', 'evaluation data': 'sample locally'}


def weighted_r2(A, target, w):
    '''R^2 of a distance-weighted linear fit - the same fit the surrogate performs'''
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A*sw[:, None], target*sw, rcond=None)
    resid = target - A @ coef
    ss_res = np.sum(w*resid**2)
    mean = np.sum(w*target)/np.sum(w)
    ss_tot = np.sum(w*(target-mean)**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


def diagnostic(clf, train_data, test_data, query_points):
    '''how linear are the black box's log-odds vs its probabilities, locally?'''
    r2_logit, r2_prob, saturation = [], [], []
    for q in query_points:
        X = get_local_points(test_data, q, samples=2000)['X']
        p = clf.predict_proba(X)[:, 1].astype(np.float64)
        w = costs.weights_based_on_distance(q, X)
        saturation.append(float(((p <= 1e-6) | (p >= 1-1e-6)).mean()))
        pc = np.clip(p, 1e-9, 1-1e-9)
        A = np.c_[X, np.ones(len(X))]
        logodds = np.log(pc/(1-pc))
        r2_logit.append(weighted_r2(A, logodds, w))
        r2_prob.append(weighted_r2(A, p, w))
    return (float(np.nanmean(r2_logit)), float(np.nanmean(r2_prob)),
            float(np.mean(saturation)))


if __name__ == '__main__':
  out = {}
  for dataset in DATASETS:
      for model in MODELS:
          key = f'{dataset}|{model}'
          entry = {'metrics': {}}
          for metric in METRICS:
              entry['metrics'][metric] = {}
              for expl in EXPLAINERS:
                  r = clime.pipeline.run_pipeline(opts(dataset, model, expl, metric),
                                                  parallel_eval=False)   # B10
                  s = np.array(r['score']['scores'])
                  entry['metrics'][metric][expl] = {'mean': float(s.mean()), 'scores': s.tolist()}
                  entry['model_stats'] = {k: float(v) for k, v in r['model_stats'].items()}
          # diagnostic, from the black box alone
          base = clime.pipeline.run_pipeline(opts(dataset, model, EXPLAINERS[0], METRICS[0]),
                                             parallel_eval=False)
          qs, _ = get_points_between_class_means(base['test_data'])
          rl, rp, sat = diagnostic(base['clf'], base['train_data'], base['test_data'], qs)
          entry['diagnostic'] = {'r2_logit': rl, 'r2_prob': rp, 'gap': rl-rp, 'saturation': sat}
          entry['eval_points'] = np.array(qs).tolist()
          out[key] = entry

          m = entry['metrics']['Brier score (local)']
          adv = m['bLIMEy (normal)']['mean']/max(m['bLIMEy (logit)']['mean'], 1e-30)
          print(f"{dataset:26s} {model:36s} gap={rl-rp:+.3f} sat={sat:5.1%} "
                f"Brier normal={m['bLIMEy (normal)']['mean']:.2e} logit={m['bLIMEy (logit)']['mean']:.2e} "
                f"advantage={adv:8.2f}x", flush=True)

  json.dump(out, open(sys.argv[1], 'w'))
  print('written', sys.argv[1])
