# Logit-LIME experiments

Everything behind the paper draft in `~/Repos/Overleaf/Logit-LIME/`
(*Logit-LIME: The Right Surrogate Depends on the Black Box*). See `FINDINGS.md` §4 in the
repo root for the summary.

## The question

LIME fits a linear model to the black box's probabilities. Probabilities are bounded,
linear models are not. Does fitting in logit space instead help? Answer: it depends on the
black box, and the dependence is measurable in advance.

## Running

```bash
conda activate clime
cd experiments/logit_lime

python experiment.py results.json   # the sweep: 4 datasets x 7 black boxes
                                    #            x 3 surrogates x 2 metrics, ~40 min
python gen_table.py                 # -> table1.tex, plus the numbers quoted in the paper
python fig_mechanism.py             # -> figs/fig1_mechanism.pdf   (needs no results.json)
python fig_diagnostic.py            # -> figs/fig2_diagnostic.pdf
python fig_spatial.py               # -> figs/fig3_spatial.pdf
```

`rerun_svm.py` re-runs only the SVM rows and merges them back into `results.json`; it was
written when the SVM's `gamma` was fixed (`FINDINGS.md` B14) and is a useful template for
patching one black box without repeating the whole sweep.

Then copy `table1.tex` and `figs/*.pdf` into the Overleaf project.

## Two things to know

**Everything runs with `parallel_eval=False`.** The local sampling draws from the global
numpy RNG, so parallel runs are not reproducible (`FINDINGS.md` B10) and the differences
that matter here are ~1e-3 — the same size as that nondeterminism. Do not switch this on
for these experiments.

**The diagnostic is the point.** `experiment.py::diagnostic` fits a locality-weighted
linear model to the black box's log-odds and to its probabilities on the same locally
sampled points, and returns the gap in weighted R². That gap predicts whether Logit-LIME
will help (Spearman rho = 0.77, n = 28) and needs only the black box, not any surrogate.

## Results shipped here

`results.json` is the full sweep behind the paper: per-configuration mean and
per-query-point scores for both metrics, the diagnostic (`r2_logit`, `r2_prob`, `gap`,
`saturation`), the black box's train/test accuracy, and the query point coordinates.

## Figure style

`style.py` holds the shared matplotlib style. The three series colours are slots 1–3 of a
validated categorical palette, used unmodified — they clear the colourblind-separation and
contrast gates as an all-pairs triple, which matters because Figure 2 is a scatter plot.
Do not substitute colours ad hoc.
