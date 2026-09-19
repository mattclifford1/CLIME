#!/bin/bash
# Regenerate every figure and table the paper uses, and copy them into the Overleaf clone.
#
# The generators read from results/ and write to figs/ and tables/ under names that match
# the \includegraphics and \input paths in the paper, so the copy is a straight mirror
# with no renaming step to get wrong.
set -eu
cd "$(dirname "$0")"

PAPER=${1:-$HOME/Repos/Overleaf/Logit-LIME}

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
         figures/fig_robustness.py figures/fig_diagnostic_blind.py; do
    echo "=== $f"
    uv run python "$f" > /dev/null
done

if [ -d "$PAPER" ]; then
    cp figs/*.pdf "$PAPER/figs/"
    cp tables/*.tex "$PAPER/tables/"
    echo "copied $(ls figs/*.pdf | wc -l) figures and $(ls tables/*.tex | wc -l) tables to $PAPER"
else
    echo "no paper at $PAPER - figures and tables left in figs/ and tables/"
fi
