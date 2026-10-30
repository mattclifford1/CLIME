'''
Local fidelity along the query-point line, with the black box's decision surface below
(clime:fig:Gaussian-all, clime:fig:UCI). Same configurations as the CIKM'23 figures.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import warnings

import matplotlib.pyplot as plt
import numpy as np

import draw
import style

import clime

WIDTH = 0.49
NORMAL, BALANCED = 'bLIMEy (normal)', 'bLIMEy (cost sensitive sampled)'
COVS = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]


def _opts(dataset, explainer, means=((-1, -1), (1, 1)), standardise=True):
    return {'standardise data': standardise,
            'data params': {'class_samples': (200, 200), 'percent of data': 0.11,
                            'moons_noise': 0.2, 'gaussian_means': [list(m) for m in means],
                            'gaussian_covs': COVS},
            'dataset': dataset, 'dataset rebalancing': 'none', 'model': 'Random Forest',
            'model balancer': 'none', 'explainer': explainer,
            'evaluation metric': 'fidelity (local)',
            'evaluation points': 'between_class_means', 'evaluation data': 'test data'}


# output name: (dataset, extra opts); means +-3 are left unstandardised, as in the paper
CONFIGS = {
    'Gaussian-1': ('Gaussian', {'means': ((-1, -1), (1, 1))}),
    'Gaussian-3': ('Gaussian', {'means': ((-3, -3), (3, 3)), 'standardise': False}),
    'Breast-Cancer': ('Breast Cancer', {}),
    'Banknote-Authentication': ('Banknote Authentication', {}),
    'Pima-Indian-Diabetes': ('Pima Indian Diabetes', {}),
}


def figure(name):
    dataset, extra = CONFIGS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        runs = {e: clime.pipeline.run_pipeline(_opts(dataset, e, **extra), parallel_eval=True)
                for e in (BALANCED, NORMAL)}
    run = runs[NORMAL]
    test, clf = run['test_data'], run['clf']
    points = np.asarray(run['score']['eval_points'])
    acc = (np.asarray(clf.predict(test['X'])) == np.asarray(test['y'])).mean()
    print(f'{name}: f accuracy on X_test {acc:.0%}')

    fig, (top, bottom) = plt.subplots(2, 1, figsize=style.figsize(WIDTH, 1.2),
                                      gridspec_kw={'height_ratios': [1, 1.5]})
    draw.fidelity_lines(top, {e: np.asarray(r['score']['scores']) for e, r in runs.items()})
    top.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)

    proj = draw.Projection(test['X'], points)
    Z, Q = proj.transform(test['X']), proj.transform(points)
    lims = draw.limits(np.vstack([Z, Q]))
    draw.surface(bottom, clf.predict_proba, proj, lims)
    draw.classes(bottom, Z, test['y'], size=7)
    draw.query_points(bottom, Q)
    bottom.set(xlabel=proj.labels[0], ylabel=proj.labels[1])
    fig.tight_layout(h_pad=0.6)
    style.save(fig, f'clime/results/combined-{name}')


def make():
    for name in CONFIGS:
        figure(name)


if __name__ == '__main__':
    make()
