'''
Freeze datasets from ~/Repos/toy_datasets into .npz files, which
clime/data/loaders/exported_npz.py registers automatically.

    uv run python sweeps/export_toy_datasets.py [output_dir]

This used to need the toy_datasets virtualenv, because clime was pinned to
scikit-learn 1.1.3 and the two could not share a process. That is no longer true: since
the 2026-08-10 upgrade toy_datasets is a declared dependency here (the `datasets` extra),
and this runs in the clime env like everything else.

The export survives the change on different grounds. Its UCI loaders fetch over the
network at load time and cache nothing, so importing them live would make a published
sweep depend on an endpoint staying up and returning identical bytes. Writing .npz is
what pins the data. **Re-running this overwrites the datasets behind
results_extended.json**, so pass an output directory if you only mean to look.

Candidates are chosen to widen the sweep along the axes that matter for the log-odds
linearity question: feature count (3 to 279) and class imbalance.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import paths

import os
import sys
import traceback
import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else str(paths.DATASETS)

CANDIDATES = [
    'XOR',                        # synthetic, deliberately non-linear
    'Habermans Breast Cancer',    # 3 features
    'Mammographic Mass',
    'Indian Liver Patient',
    'Heart Failure',
    'Heart Disease',
    'Breast Cancer Coimbra',
    'Parkinsons',
    'Chronic Kidney Disease',
    'Thyroid Sick',               # strongly imbalanced
    'SPECTF Heart',
    'Z-Alizadeh Sani CAD',
    'Stroke Prediction',
    'Thoracic Surgery',
    'Arrhythmia',                 # 279 features
    # added 2026-09-19 (fifth registration): every remaining tabular set in toy_datasets
    # that is not already in the grid. Breast Cancer Wisconsin is sklearn's Breast Cancer
    # again and the costcla / Pima / Ionosphere etc. loaders duplicate CLIME's own
    'Breast Cancer Prognostic',
    'Cervical Cancer',            # 6.4% minority
    'Framingham CHD',             # 4240 rows
    'German Credit',
    'HCC Survival',               # 49 features on 165 rows
    'Hepatitis',
]

# Synthetic families, generated here rather than by toy_datasets. Its GaussianGenerator
# re-seeds before drawing each class, so class 1 comes out as an exact translate of class
# 0, point for point - fine for a picture, wrong for anything that fits to the joint
# sample. The parameters are in the name, and so in the registry key and every plot label.
#
# Gaussian: two classes, identity covariance for class 0 and r * identity for class 1, so
# the Bayes log-odds are exactly linear when r = 1 and exactly quadratic when r = 3. The
# means sit at -/+ (s/2) along the diagonal, *normalised*, so s is the Euclidean distance
# between them whatever d is (toy_datasets offsets every coordinate by s, which makes the
# separation grow as sqrt(d) and confounds the two).
GAUSS_D = [2, 5, 10, 30]
GAUSS_R = [1, 3]
GAUSS_S = [2, 4, 6]
GAUSS_N = 1000
# make_classification: one Gaussian cluster per class on a hypercube of informative
# features, the rest pure noise. Five informative features at every d, and half of d, to
# separate "many features" from "many relevant features"
MAKECLF = [(10, 5), (30, 5), (30, 15), (60, 5), (60, 30),
           (100, 5), (100, 50), (200, 5), (200, 100)]
MAKECLF_N = 2000
SYNTH_SEED = 20260919

MAX_FEATURES = 300
MIN_PER_CLASS = 25


def main():
    from data_loaders.main import get_dataset
    os.makedirs(OUT_DIR, exist_ok=True)
    kept, skipped = [], []

    for name in CANDIDATES:
        try:
            loader = get_dataset(name)
            train, test = loader.get_train_test_split()
            X = np.asarray(np.concatenate([train['X'], test['X']]), dtype=np.float64)
            y = np.asarray(np.concatenate([train['y'], test['y']])).ravel()

            classes, counts = np.unique(y, return_counts=True)
            if len(classes) != 2:
                skipped.append((name, f'{len(classes)} classes, not binary'))
                continue
            if X.shape[1] > MAX_FEATURES:
                skipped.append((name, f'{X.shape[1]} features > {MAX_FEATURES}'))
                continue
            if counts.min() < MIN_PER_CLASS:
                skipped.append((name, f'smallest class {counts.min()} < {MIN_PER_CLASS}'))
                continue
            if not np.isfinite(X).all():
                # a few of the medical sets carry NaNs for missing values; median impute
                # so the comparison is about the surrogate, not about imputation choices
                for j in range(X.shape[1]):
                    col = X[:, j]
                    bad = ~np.isfinite(col)
                    if bad.any():
                        col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
            # relabel to 0/1 in case the loader uses other codes
            y = (y == classes[1]).astype(np.int64)

            try:
                names = list(loader.get_feature_names())
            except Exception:
                names = [f'feature {i}' for i in range(X.shape[1])]
            if len(names) != X.shape[1]:
                names = [f'feature {i}' for i in range(X.shape[1])]

            path = os.path.join(OUT_DIR, name.replace(' ', '_') + '.npz')
            # dtype=str, not object: an object array is pickled, and the pickle carries a
            # numpy 2.x module path that the numpy 1.24 in the clime env cannot import
            np.savez_compressed(path, X=X, y=y, feature_names=np.array(names, dtype=str))
            kept.append((name, X.shape[0], X.shape[1], counts.min()/counts.sum()))
            print(f'  kept    {name:28s} n={X.shape[0]:5d} d={X.shape[1]:4d} '
                  f'minority={counts.min()/counts.sum():.1%}')
        except Exception as e:
            skipped.append((name, f'{type(e).__name__}: {e}'))
            print(f'  FAILED  {name:28s} {type(e).__name__}: {e}', file=sys.stderr)
            traceback.print_exc(limit=1, file=sys.stderr)

    for name, X, y in synthetic():
        path = os.path.join(OUT_DIR, name.replace(' ', '_') + '.npz')
        names = np.array([f'x{i+1}' for i in range(X.shape[1])], dtype=str)
        np.savez_compressed(path, X=X, y=y, feature_names=names)
        counts = np.bincount(y)
        kept.append((name, X.shape[0], X.shape[1], counts.min()/counts.sum()))
        print(f'  made    {name:28s} n={X.shape[0]:5d} d={X.shape[1]:4d} '
              f'minority={counts.min()/counts.sum():.1%}')

    print(f'\n{len(kept)} exported to {OUT_DIR}')
    for name, why in skipped:
        print(f'  skipped {name:28s} {why}')


def synthetic():
    '''(name, X, y) for every member of the two synthetic families'''
    from sklearn.datasets import make_classification
    for d in GAUSS_D:
        for r in GAUSS_R:
            for sep in GAUSS_S:
                # one generator per dataset, seeded from its parameters, so adding a
                # member to the family never changes the others
                rng = np.random.default_rng([SYNTH_SEED, d, r, sep])
                offset = np.ones(d)/np.sqrt(d)*sep/2
                n = GAUSS_N//2
                X0 = rng.multivariate_normal(-offset, np.eye(d), n)
                X1 = rng.multivariate_normal(+offset, r*np.eye(d), n)
                X = np.vstack([X0, X1])
                y = np.r_[np.zeros(n), np.ones(n)].astype(np.int64)
                yield f'Gauss d{d} r{r} s{sep}', X, y
    for d, i in MAKECLF:
        X, y = make_classification(n_samples=MAKECLF_N, n_features=d, n_informative=i,
                                   n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                                   class_sep=1.0, flip_y=0.01, shuffle=True,
                                   random_state=SYNTH_SEED + d*1000 + i)
        yield f'MakeClf d{d} i{i}', X.astype(np.float64), y.astype(np.int64)


if __name__ == '__main__':
    main()
