'''
Export datasets from ~/Repos/toy_datasets into .npz files CLIME can read.

Run this with the toy_datasets virtualenv, NOT the clime env:

    ~/Repos/toy_datasets/.venv/bin/python export_toy_datasets.py

toy_datasets requires numpy>=2.3.5 and scikit-learn>=1.7.2; clime is pinned to
scikit-learn 1.1.3 and upgrading it silently changes every result in this repo
(CLAUDE.md). The two cannot share a process, so we cross the boundary with data on
disk rather than by importing code.

Candidates are chosen to widen the sweep along the axes that matter for the log-odds
linearity question: feature count (3 to 279) and class imbalance.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import sys
import traceback
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_datasets')

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
]

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

    print(f'\n{len(kept)} exported to {OUT_DIR}')
    for name, why in skipped:
        print(f'  skipped {name:28s} {why}')


if __name__ == '__main__':
    main()
