import matplotlib.pyplot as plt
import clime
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--single', '-s', action='store_false')
args=parser.parse_args()
# logging.basicConfig(level=logging.INFO)
opts = {
    # 'dataset':             'credit scoring 1',
    'dataset':             'moons',
    'dataset':             'Gaussian',
    'data params': {'class_samples':  [5, 75], # only for syntheic datasets
                    'percent_of_data': 0.05,    # for real datasets
                    'moons_noise': 0.2,
                    'gaussian_means': [[1, 0], [1, 1]],
                    'gaussian_covs': [[[0.5, 0], [0, 0.5]],    [[0.5, 0], [0, 0.5]]],
                    },
    'dataset rebalancing': 'none',
    # 'model':               'SVM',
    'model':               'Random Forest',
    # 'model':               'Bayes Optimal',
    'model balancer':      'none',
    'explainer':           'bLIMEy (cost sensitive sampled)',
    # 'explainer':           'bLIMEy (cost sensitive class)',
    # 'explainer':           'bLIMEy (normal)',
    # 'explainer':         'LIME (original)',
    'evaluation metric': 'fidelity (local)',
    # 'evaluation metric':   'fidelity (normal)',
    'evaluation points': 'between_class_means',
    'evaluation data': 'test data'
}

p = clime.pipeline.construct(opts)
result = p.run(parallel_eval=args.single)
scores = result['score']['scores']
x = list(range(len(scores)))
plt.plot(x, scores)
plt.show()
