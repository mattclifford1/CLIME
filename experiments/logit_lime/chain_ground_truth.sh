#!/bin/bash
# run last: waits for the other sweeps so three jobs do not contend for the same cores
source /home/matt/anaconda3/etc/profile.d/conda.sh && conda activate clime
cd "$(dirname "$0")" || exit 1
while pgrep -f "[s]weep_extended\.py|[s]weep_explanations" > /dev/null; do sleep 120; done
python -u sweep_ground_truth.py results_ground_truth.json >> sweep_ground_truth.log 2>&1
echo "GROUND TRUTH SWEEP DONE"
