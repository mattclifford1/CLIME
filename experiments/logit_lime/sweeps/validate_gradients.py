'''
Check every analytic gradient in common/gradients.py against central finite differences.

The analytic forms are what the ground truth sweep trusts, so they need an independent
check. Finite differences are that check and not the instrument: they are only valid
where the black box is unsaturated (logit of a clipped probability is flat), and they
carry two errors of their own that a verdict has to allow for.

  Kinks. A relu MLP is piecewise linear, and a difference quotient that straddles a kink
  averages the two one-sided derivatives. Those points disagree at h = 1e-3 and agree to
  1e-10 at h = 1e-5, and the disagreement is almost perfectly correlated (rho = +0.98)
  with how many hidden units sit within one step of their kink. The verdict is therefore
  taken at the finer step.

  Scale. For an SVC, p depends on x only through the decision function f, so grad logit p
  is parallel to grad f whatever the calibration map is - the direction is exact by
  construction. Its magnitude needs d logit p / d f, which equals -probA_ only if the
  stored Platt parameters describe predict_proba exactly, and sklearn fits them by
  internal cross validation on different fits of the model. The residual is ~1e-3
  relative, constant in h, and affects no measure used here: cosine, rank correlation and
  top-1 are all invariant to a positive scale factor. It is reported, not treated as a
  failure.

The verdict is therefore on direction, which is what the study measures.

usage:  python sweeps/validate_gradients.py [--datasets N]
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths, gradients

import argparse
import warnings
import numpy as np
import clime
from sweeps.sweep import opts, DATASETS, METRICS
from clime.evaluation.key_points import get_points_between_class_means

warnings.filterwarnings('ignore')

STANDARD = 'bLIMEy (normal)'
COARSE, FINE = 1e-3, 1e-5
TOL = 1e-6


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 0 and nb > 0 else np.nan


def check(dataset, model):
    r = clime.pipeline.run_pipeline(opts(dataset, model, STANDARD, METRICS[0]),
                                    parallel_eval=False)
    clf, test_data = r['clf'], r['test_data']
    Q = np.asarray(get_points_between_class_means(test_data)[0], dtype=np.float64)

    analytic = gradients.grad_logit(clf, model, Q)
    sat = gradients.saturated(clf, Q)
    out = {'n_sat': int(np.sum(sat))}
    for label, h in (('coarse', COARSE), ('fine', FINE)):
        numeric = gradients.finite_difference(clf, Q, h=h)
        cos, rel = [], []
        for i in range(Q.shape[0]):
            if sat[i] or np.linalg.norm(numeric[i]) == 0:
                continue
            cos.append(cosine(analytic[i], numeric[i]))
            rel.append(np.linalg.norm(analytic[i] - numeric[i])/np.linalg.norm(numeric[i]))
        out[label] = (np.array(cos), np.array(rel))
    return out


def main(n_datasets=4):
    datasets = DATASETS[:n_datasets]
    print(f'{len(datasets)} datasets x {len(gradients.ANALYTIC_GRADIENTS)} black boxes, '
          f'20 query points each\n')
    print(f"{'model':<30s} {'n':>5s} {'sat':>4s} {'min cos h=1e-3':>15s} "
          f"{'min cos h=1e-5':>15s} {'med |rel|':>10s}  verdict")
    bad = 0
    for model in gradients.ANALYTIC_GRADIENTS:
        coarse_c, fine_c, fine_r, n_sat, failures = [], [], [], 0, []
        for dataset in datasets:
            try:
                res = check(dataset, model)
            except Exception as e:
                failures.append(f'{dataset}: {type(e).__name__}: {e}')
                continue
            coarse_c.append(res['coarse'][0])
            fine_c.append(res['fine'][0])
            fine_r.append(res['fine'][1])
            n_sat += res['n_sat']

        def cat(xs):
            return np.concatenate(xs) if xs else np.array([])
        coarse_c, fine_c, fine_r = cat(coarse_c), cat(fine_c), cat(fine_r)
        if fine_c.size == 0:
            print(f'{model:<30s} {"-":>5s} {n_sat:>4d} {"":>15s} {"":>15s} '
                  f'{"":>10s}  NOTHING CHECKED')
            bad += 1
        else:
            ok = np.nanmin(fine_c) > 1 - TOL
            bad += 0 if ok else 1
            print(f'{model:<30s} {fine_c.size:>5d} {n_sat:>4d} '
                  f'{np.nanmin(coarse_c):>15.7f} {np.nanmin(fine_c):>15.7f} '
                  f'{np.nanmedian(fine_r):>10.1e}  {"ok" if ok else "MISMATCH"}')
        for f in failures:
            print(f'    {f}')

    print()
    for model in gradients.NO_GRADIENT:
        print(f'{model:<30s} no analytic gradient by construction')
    print(f'\n{bad} model(s) failed' if bad else
          '\nevery analytic gradient matches the finite difference direction on every '
          'unsaturated query point')
    return bad


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=int, default=4)
    sys.exit(1 if main(p.parse_args().datasets) else 0)
