#!/bin/bash
source /home/matt/anaconda3/etc/profile.d/conda.sh && conda activate clime
cd /home/matt/projects/CLIME/experiments/logit_lime || exit 1
# wait on the PYTHON pids, not the bash wrappers that launched them, and not on name
# patterns (a pattern also matches any monitor whose command line contains it)
while [ -d /proc/3551854 ] || [ -d /proc/3749912 ]; do sleep 120; done
python -u sweep_ground_truth.py results_ground_truth.json >> sweep_ground_truth.log 2>&1
echo "GROUND TRUTH SWEEP DONE"
