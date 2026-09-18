#!/bin/bash
# Re-run every sweep on the upgraded stack (python 3.13, scikit-learn 1.9, numpy 2.4).
#
# The previous results were produced on scikit-learn 1.1.3 / numpy 1.24 and are archived
# under results/archive/sklearn1.1.3/ rather than deleted: the point of re-running is to
# find out how much the upgrade moved the numbers, which needs both sets.
#
# Every sweep resumes from its output file, so this is safe to re-run after an
# interruption. Order matters only for machine load, not correctness.
set -u
cd "$(dirname "$0")" || exit 1

ARCHIVE=results/archive/sklearn1.1.3
if [ ! -d "$ARCHIVE" ]; then
    mkdir -p "$ARCHIVE"
    cp results/results_*.json "$ARCHIVE"/ 2>/dev/null
    echo "archived $(ls "$ARCHIVE" | wc -l) result files to $ARCHIVE/"
fi

# start each sweep from scratch on the new stack - resuming into a file written by the
# old stack would silently mix the two
rm -f results/results_taxonomy.json results/results_extended.json \
      results/results_kernel.json results/results_explanations.json \
      results/results_explanations_extended.json results/results_ground_truth.json \
      results/results_gradient_truth.json results/results_taylor.json \
      results/results_fidelity.json results/results_seed*.json \
      results/results_fidelity_extended.json results/results_instruments.json \
      results/results_null.json results/results_range.json

run () {
    echo "=== $1 ==="
    uv run python -u "$@" || { echo "FAILED: $1"; exit 1; }
}

run sweeps/sweep.py results_taxonomy.json
run sweeps/sweep_extended.py results_extended.json
run sweeps/sweep_kernel.py results_kernel.json
run sweeps/sweep_explanations.py results_explanations.json
run sweeps/sweep_explanations_extended.py results_explanations_extended.json
run sweeps/sweep_ground_truth.py results_ground_truth.json
# the analytic gradients the next sweep trusts, checked against finite differences first:
# a wrong closed form would silently corrupt every explanation score downstream
run sweeps/validate_gradients.py
run sweeps/sweep_gradient_truth.py results_gradient_truth.json
run sweeps/sweep_taylor.py results_taylor.json
run sweeps/sweep_patches.py results_patches.json
run sweeps/sweep_fidelity.py results_fidelity.json
# the same 2x2 for the five extended-grid black boxes that have a gradient ground truth,
# into its own file so the registered 168 above stays exactly that
run sweeps/sweep_fidelity.py results_fidelity_extended.json \
    --models 'Bagged Logistic,Bayes Optimal,Nearest Class Mean,Polynomial Logistic (deg 2),RBF Logistic (Nystroem)'
run sweeps/sweep_instruments.py results_instruments.json
run sweeps/sweep_null.py results_null.json
run sweeps/sweep_range.py results_range.json
run sweeps/sweep_seeds.py results

echo "=== ALL SWEEPS COMPLETE ==="
