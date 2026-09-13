#!/bin/bash
# Every aLIMEgn sweep, in order, with the thread setting they need.
#
# Each sweep caches one file per configuration, so this is safe to interrupt and re-run:
# it picks up whatever is missing. To force a recompute, delete results/cache/<sweep>/.
set -eu
cd "$(dirname "$0")"

PROCESSES=${1:-16}

# sklearn's BLAS threads and the worker processes otherwise oversubscribe the machine
export OMP_NUM_THREADS=1

mkdir -p logs

echo "=== marginal: the evaluation marginal, and what weighting does about it (P1, P2, P5)"
uv run python sweeps/sweep_marginal.py "$PROCESSES" 2>&1 | tee logs/marginal.log

echo "=== degrade: force P(yhat|x) away from P(y|x) (P3, P4, P6)"
uv run python sweeps/sweep_degrade.py "$PROCESSES" --seeds 1,2 2>&1 | tee logs/degrade.log

echo "=== grid: where in the space the mismatch lives"
uv run python sweeps/sweep_grid.py "$((PROCESSES/2))" 2>&1 | tee logs/grid.log

echo
echo "=== findings"
uv run python analysis/analyse_marginal.py 2>&1 | tee logs/analyse_marginal.log
uv run python analysis/analyse_degrade.py 2>&1 | tee logs/analyse_degrade.log
