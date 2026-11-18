# CIKM'23, extended

The CIKM'23 protocol (LIME against class-balanced LIME, scored by local fidelity along the line through the class means) on 22 datasets, six model families and three class ratios, with the black box's and the surrogate's thresholds as extra arms.

Write-up: `../../../papers/CIKM-2023-extra-datasets`.

- **Data** (`common/tasks.py`): each dataset at its own class ratio (70:30 split), balanced at 110:110 and imbalanced at 200:20, plus ratios 4, 20 and 40 for the sweep.
- **Black boxes** (`common/boxes.py`): the six `projection_models` families of `clime/models/projection.py`, trained plain, class weighted or on SMOTE-balanced data.
- **Threshold of the black box** (`clime/models/threshold.py`): 0.5, re-thresholded on validation data, or placed by each named deltas method on the training projections.
- **Surrogates** (`common/surrogates.py`): LIME, CLIME, and LIME's surrogate with its 0.5 cut moved by balanced accuracy or by a deltas method.
- **Scores** (`common/scores.py`): fidelity and per-class fidelity (operating point), AUC (ranking) and Brier score (calibration) against the black box, on the test set and on a local sample.
- **Explanations** (`common/explanations.py`): six surrogates on one neighbourhood sample, scored against the black box's gradient, across two draws of the neighbourhood and across each correction of the black box; predictions in `PREREGISTRATION.md`.

## Running

`N_JOBS` sets the number of workers (default 16). Every run caches one file per job under `results/raw/`, so an interrupted run resumes.

```bash
uv sync --extra notebook --extra datasets --extra figures --extra thesis --group dev
cd experiments/cikm_extended
uv run pytest test_cikm_extended.py
uv run python run_reproduce.py          # the five published configurations, through clime's pipeline
uv run python run_main.py               # 3 class ratios x 6 families x 10 seeds, every arm
uv run python run_sweep.py              # imbalance ratios 4, 20 and 40
uv run python run_kernel.py             # locality kernel width
uv run python run_surrogate_deltas.py   # every deltas method on the surrogate's threshold
uv run python run_grid.py               # fidelity maps on the first two principal components
uv run python run_matched.py            # CLIME's gain split into a change of cut and a change of fit
uv run python run_explanations.py       # six surrogates scored as explanations, 5 seeds
./make_paper.sh                         # summarise, draw figures and tables, copy them into the paper
```

`./rerun_all.sh` runs the four large runs in stages, three seeds of the main run first.
`check_outliers.py` repeats the headline comparison without the splits whose test set holds an extreme value.
