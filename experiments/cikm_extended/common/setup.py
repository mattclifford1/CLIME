'''import first in every run script: one thread per worker, quiet progress bars'''
import os
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('TQDM_DISABLE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
