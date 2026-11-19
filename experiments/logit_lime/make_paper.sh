#!/bin/bash
# Regenerate every figure and table, and copy the ones each paper uses into its Overleaf clone.
#
# usage: ./make_paper.sh [paper directory ...]
set -eu
cd "$(dirname "$0")"

PAPERS=${*:-"$HOME/Repos/clime/papers/Logit-LIME-method $HOME/Repos/clime/papers/evaluation-of-explanations"}

# tables that take arguments, run before the plain loop below
echo "=== analysis/table_fidelity_proxy.py --extended"
uv run python analysis/table_fidelity_proxy.py --extended > /dev/null

for f in figures/fig_mechanism.py figures/fig_diagnostic.py figures/fig_spatial.py \
         figures/fig_groups.py figures/fig_kernel.py figures/fig_setup.py \
         figures/fig_justification.py figures/fig_group_gallery.py figures/fig_digits.py \
         figures/fig_patches.py \
         figures/fig_fidelity_explanation.py figures/fig_taylor_tradeoff.py \
         figures/fig_instruments.py figures/fig_blind.py figures/fig_reading.py \
         analysis/table_brier_kl.py analysis/table_groups.py analysis/table_fidelity.py \
         analysis/table_example_explanation.py analysis/table_gradient_truth.py \
         analysis/table_instruments.py analysis/table_reading.py \
         analysis/analyse_range.py \
         analysis/analyse_diagnostic.py analysis/analyse_robustness.py \
         figures/fig_robustness.py figures/fig_diagnostic_blind.py \
         analysis/analyse_soft_logistic.py figures/fig_softlabel.py; do
    echo "=== $f"
    uv run python "$f" > /dev/null
done

for PAPER in $PAPERS; do
    if [ ! -d "$PAPER" ]; then
        echo "no paper at $PAPER"
        continue
    fi
    n=0
    for f in figs/*.pdf tables/*.tex; do
        if grep -rqF "${f%.tex}}" "$PAPER/main.tex" "$PAPER/sections"; then
            cp "$f" "$PAPER/$f"
            n=$((n + 1))
        fi
    done
    echo "copied $n figures and tables to $PAPER"
done
