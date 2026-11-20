'''The two-feature toys of the threshold and sample-count figures: two Gaussian classes, a model fitted to them.'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import clime
from sweeps.sweep import opts, DATA_PARAMS, METRICS
from clime.evaluation.key_points import get_points_between_class_means

IDENTITY = [[1, 0], [0, 1]]
# x0 separates the classes about three times as well as x1
LINEAR = dict(model='Logistic', gaussian_means=[[-1, -0.3], [1, 0.3]],
              gaussian_covs=[IDENTITY, IDENTITY])
# each class is narrow along a different feature, so the fitted log-odds are quadratic
CURVED = dict(model='QDA', gaussian_means=[[-1, -1], [1, 1]],
              gaussian_covs=[[[1, 0], [0, 0.3]], [[0.3, 0], [0, 1]]])


def build(model, **data):
    '''(black box, train data, test data, the 20 query points between the class means)'''
    o = opts('Gaussian', model, 'bLIMEy (normal)', METRICS[0])
    o['data params'] = {**DATA_PARAMS, **data}
    r = clime.pipeline.run_pipeline(o, parallel_eval=False)
    qs = np.asarray(get_points_between_class_means(r['test_data'])[0], dtype=float)
    return r['clf'], r['train_data'], r['test_data'], qs
