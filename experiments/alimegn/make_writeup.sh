#!/bin/bash
# Regenerate every aLIMEgn figure and table, and copy them into the Overleaf clones.
#
# usage: ./make_writeup.sh [write-up directory] [CIKM extension paper directory]
set -eu
cd "$(dirname "$0")"

WRITEUP=${1:-$HOME/Repos/clime/papers/aLIMEgn}
EXTENDED=${2:-$HOME/Repos/clime/papers/CIKM-2023-extra-datasets}

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

# the CIKM extension paper holds the same results as an appendix, with its own table labels
if [ -d "$EXTENDED" ]; then
    mkdir -p "$EXTENDED/figures/alimegn" "$EXTENDED/tables/alimegn"
    cp figs/*.pdf "$EXTENDED/figures/alimegn/"
    for f in tables/*.tex; do
        sed 's/\\label{tab:/\\label{tab:alimegn-/' "$f" > "$EXTENDED/tables/alimegn/$(basename "$f")"
    done
    echo "copied the same to $EXTENDED"
fi
