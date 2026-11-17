#!/bin/bash
# Every run, in stages so that a first pass over all of them finishes early: three seeds of
# the main run and one of each other run, then the rest. Finished jobs are skipped.
set -u
cd "$(dirname "$0")"
mkdir -p logs
run() { N_JOBS=${N_JOBS:-16} nice -n 5 uv run python "run_$1.py" ${2:-} >> "logs/$1.log" 2>&1; }
run main 3
run kernel 1
run sweep 1
run surrogate_deltas 1
run main
run sweep
run kernel
run surrogate_deltas
