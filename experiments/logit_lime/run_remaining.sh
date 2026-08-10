#!/bin/bash
# Run the three sweeps in turn. All three resume from their output JSON, so this is safe
# to re-run after an interruption - it picks up at the first configuration not yet done.
source /home/matt/anaconda3/etc/profile.d/conda.sh && conda activate clime
cd "$(dirname "$0")" || exit 1

echo "=== taxonomy sweep ==="
python sweep.py results_taxonomy.json >> sweep_taxonomy.log 2>&1 || exit 1
echo "=== kernel sweep ==="
python sweep_kernel.py results_kernel.json >> sweep_kernel.log 2>&1 || exit 1
echo "=== seed sweep ==="
python sweep_seeds.py results >> sweep_seeds.log 2>&1 || exit 1
echo "=== ALL SWEEPS COMPLETE ==="
