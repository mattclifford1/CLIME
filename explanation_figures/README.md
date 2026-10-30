# `explanation_figures`: CLIME thesis figures in one style

The CLIME and Logit-LIME thesis figures, regenerated from code in the same style as
`deltas/linear_confidence/explanation_figures`.

```bash
uv run python explanation_figures/make_all.py                 # all figures
uv run python explanation_figures/make_all.py cikm_fidelity   # one module
uv run python explanation_figures/make_all.py --thesis        # all, then copy changed ones into the thesis
uv run python explanation_figures/sync_thesis.py --dry-run    # list what a copy would change
```

PDFs go to `out/<chapter>/...`, mirroring the thesis's `chapters/<chapter>/figures/...`.
`sync_thesis.py` copies changed files across (set `THESIS_DIR` to point elsewhere); committing the thesis stays manual.

| Module | Output (under `out/`) | Thesis label |
|---|---|---|
| `lime_diagram.py` | `02-literature-review/LIME` | `fig:literature-review:lime` |
| `cikm_fidelity.py` | `clime/results/combined-{Gaussian-1,Gaussian-3,Breast-Cancer,Banknote-Authentication,Pima-Indian-Diabetes}` | `clime:fig:Gaussian-all`, `clime:fig:UCI` |
| `alimegn.py` | `clime/extensions/fig_schematic`, `fig_marginal`, `fig_grid_Gaussian` | `clime:fig:alimegn-schematic`, `-marginal`, `-grid` |

Change the shared look in `style.py` and the shared drawing pieces in `draw.py`.
Figures are drawn at printed size, so include each at the `WIDTH` set in its module
(`\includegraphics[width=<WIDTH>\textwidth]{...}`); `alimegn.py` figures are full width.

## Adding a figure

Write a module with a `make()` that draws with `style` and `draw` and ends in `style.save(fig, '<chapter>/<name>')`,
then add it to `FIGURES` in `make_all.py` and to the table above.

## Differences from the published figures

- Class 0 is blue and class 1 red, matching the deltas figures and the CIKM text's region colours.
- The query-point line runs from class 0 to class 1 (`FINDINGS.md` B7), so Breast Cancer is mirrored relative to the paper.
- Scores come from the current pipeline (per-point seeding, B10), so they differ from the paper's in the third decimal.
- `LIME` is drawn from a fitted black box and a weighted ridge surrogate instead of by hand.
- The aLIMEgn figures read the same cached results as `experiments/alimegn/figures/`, without suptitles.
