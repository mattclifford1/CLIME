'''
The five configurations of the CIKM'23 figures, rerun through clime's pipeline with the
options of experiments/lime_vs_clime-sampling.py. Writes results/reproduce.csv.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import pandas as pd
import clime
from common import paths

EXPLAINERS = {'LIME': 'bLIMEy (normal)', 'CLIME': 'bLIMEy (cost sensitive sampled)'}
# name -> (dataset, class means, standardised). The published distant Gaussians were left
# unstandardised (experiments/gaussian_lime_vs_clime.py); the last row standardises them.
CONFIGURATIONS = {
    'Gaussian, means at ±1': ('Gaussian', [[-1, -1], [1, 1]], True),
    'Gaussian, means at ±3': ('Gaussian', [[-3, -3], [3, 3]], False),
    'Breast Cancer': ('Breast Cancer', None, True),
    'Banknote Authentication': ('Banknote Authentication', None, True),
    'Pima Indian Diabetes': ('Pima Indian Diabetes', None, True),
    'Gaussian, means at ±3, standardised': ('Gaussian', [[-3, -3], [3, 3]], True),
}


def options(dataset, means, standardise, explainer):
    return {'data params': {'class_samples': (200, 200), 'percent of data': 0.11,
                            'moons_noise': 0.2,
                            'gaussian_means': means or [[-1, -1], [1, 1]],
                            'gaussian_covs': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]},
            'dataset': dataset, 'standardise data': standardise,
            'dataset rebalancing': 'none', 'model': 'Random Forest',
            'model balancer': 'none', 'explainer': explainer,
            'evaluation metric': 'fidelity (local)',
            'evaluation points': 'between_class_means', 'evaluation data': 'test data'}


if __name__ == '__main__':
    rows = []
    for name, (dataset, means, standardise) in CONFIGURATIONS.items():
        for label, explainer in EXPLAINERS.items():
            result = clime.pipeline.run_pipeline(options(dataset, means, standardise, explainer))
            accuracy = result['model_stats']
            for i, score in enumerate(result['score']['scores']):
                rows.append({'configuration': name, 'explainer': label, 'index': i,
                             'fidelity': float(score),
                             **{k: float(v) for k, v in accuracy.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(paths.out(paths.RESULTS, 'reproduce.csv'), index=False)
    print(frame.groupby(['configuration', 'explainer'], sort=False)['fidelity']
          .agg(['mean', 'min']).round(6).to_string())
