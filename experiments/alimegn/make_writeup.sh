#!/bin/bash
# Regenerate every figure and table the aLIMEgn write-up uses, and copy them into the
# Overleaf clone. Generated names match the write-up's \includegraphics and \input paths,
# so the copy is a mirror with no renaming step to get wrong.
set -eu
cd "$(dirname "$0")"

WRITEUP=${1:-$HOME/Repos/Overleaf/aLIMEgn}

for f in figures/fig_schematic.py figures/fig_marginal.py figures/fig_degrade.py \
         figures/fig_weights.py "figures/fig_grid.py Gaussian" "figures/fig_grid.py Moons" \
         analysis/table_predictions.py analysis/table_schemes.py \
         analysis/analyse_balance.py; do
    echo "=== $f"
    # shellcheck disable=SC2086  # the grid figure takes a dataset argument
    uv run python $f > /dev/null
done

if [ -d "$WRITEUP" ]; then
    mkdir -p "$WRITEUP/figs" "$WRITEUP/tables"
    cp figs/*.pdf "$WRITEUP/figs/"
    cp tables/*.tex "$WRITEUP/tables/"
    echo "copied $(ls figs/*.pdf | wc -l) figures and $(ls tables/*.tex | wc -l) tables to $WRITEUP"
else
    echo "no write-up at $WRITEUP - figures and tables left in figs/ and tables/"
fi
