'''
Fidelity of LIME and CLIME over a grid on the first two principal components, for the plain
random forest. Writes results/grid.csv (one row per grid point and explainer) and
results/grid_test.csv (the test points in the same plane).
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from common import setup   # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import clime
from common import boxes, evaluate, paths, runner, tasks

CONFIGURATIONS = [('Breast Cancer Wisconsin', 'native'), ('Diabetes Pima Indian', 'native'),
                  ('Banknote Authentication', 'native'), ('Diabetes Pima Indian', 'imbalanced'),
                  ('Banknote Authentication', 'imbalanced'), ('Phoneme', 'imbalanced')]
FAMILY = 'Random Forest'
SIDE = 20
SEED = 0


def job(dataset, setting):
    clime.RANDOM_SEED = SEED
    task = tasks.load(dataset, setting, SEED)
    built, _ = boxes.build(task, FAMILY, arms=('plain',), deltas=())
    box = built['plain']
    pca = PCA(n_components=2, svd_solver='full').fit(task.train['X'])
    plane = pca.transform(task.test['X'])
    axes = [np.linspace(plane[:, i].min(), plane[:, i].max(), SIDE) for i in (0, 1)]
    tags = {'dataset': dataset, 'setting': setting}
    rows = []
    for u in axes[0]:
        for v in axes[1]:
            q = pca.inverse_transform([[u, v]])[0]
            query, scored = evaluate.explain(box, task, q, deltas=())
            rows += [{**tags, 'pc1': u, 'pc2': v, **query, **r} for r in scored]
            box.clear()
    test = pd.DataFrame({**tags, 'pc1': plane[:, 0], 'pc2': plane[:, 1], 'y': task.test['y'],
                         'f': box.predict_proba(task.test['X'])[:, 1]})
    return pd.DataFrame(rows), test


if __name__ == '__main__':
    done = Parallel(n_jobs=min(runner.N_JOBS, len(CONFIGURATIONS)))(
        delayed(job)(*c) for c in CONFIGURATIONS)
    pd.concat([d[0] for d in done]).to_csv(paths.out(paths.RESULTS, 'grid.csv'), index=False)
    pd.concat([d[1] for d in done]).to_csv(paths.out(paths.RESULTS, 'grid_test.csv'), index=False)
