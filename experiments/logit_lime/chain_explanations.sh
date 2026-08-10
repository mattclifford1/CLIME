#!/bin/bash
# wait for the base explanation sweep, then extend it to the new datasets and models
source /home/matt/anaconda3/etc/profile.d/conda.sh && conda activate clime
cd "$(dirname "$0")" || exit 1
while pgrep -f "[s]weep_explanations\.py" > /dev/null; do sleep 60; done
python -u sweep_explanations_extended.py results_explanations_extended.json \
    >> sweep_expl_extended.log 2>&1
echo "EXTENDED EXPLANATION SWEEP DONE"
