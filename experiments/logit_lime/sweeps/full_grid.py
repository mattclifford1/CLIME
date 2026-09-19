'''
The full exploratory grid of the fifth registration: 71 datasets x 16 black boxes.

One definition, imported by every sweep that runs on it and by the analysis, so the grid
cannot drift between experiments. It is a superset of the extended grid
(results_extended.json, 29 x 16) - which stays as cited in the paper - and of the
registered grid (results_taxonomy.json, 14 x 12), which stays frozen.

Families, for reporting (the registration never pools a seen family into a test):

    registered   the 14 datasets fixed in the first registration
    extended     the 15 exported from toy_datasets for results_extended.json
    new real     9 real datasets never swept before
    gaussian     24 two-Gaussian datasets: d x covariance ratio r x separation s
    makeclf      9 make_classification datasets: d x informative features

`configure()` points the module-level lists in sweep.py (and the sweeps that import from
it) at this grid, which is how the existing sweeps are reused unchanged.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import gradients
from sweeps import sweep
from sweeps.sweep_extended import NEW_MODELS, NEW_GROUPS

REGISTERED = list(sweep.DATASETS)
EXTENDED = ['Arrhythmia', 'Breast Cancer Coimbra', 'Chronic Kidney Disease',
            'Habermans Breast Cancer', 'Heart Disease', 'Heart Failure',
            'Indian Liver Patient', 'Mammographic Mass', 'Parkinsons', 'SPECTF Heart',
            'Stroke Prediction', 'Thoracic Surgery', 'Thyroid Sick', 'XOR',
            'Z-Alizadeh Sani CAD']
NEW_REAL = ['Breast Cancer Prognostic', 'Cervical Cancer', 'Framingham CHD',
            'German Credit', 'HCC Survival', 'Hepatitis',
            'Credit Scoring 2', 'Blobs', 'Digits 3 vs 8']
GAUSSIAN = [f'Gauss d{d} r{r} s{s}' for d in (2, 5, 10, 30) for r in (1, 3) for s in (2, 4, 6)]
MAKECLF = [f'MakeClf d{d} i{i}' for d, i in [(10, 5), (30, 5), (30, 15), (60, 5), (60, 30),
                                             (100, 5), (100, 50), (200, 5), (200, 100)]]

FAMILIES = {'registered': REGISTERED, 'extended': EXTENDED, 'new real': NEW_REAL,
            'gaussian': GAUSSIAN, 'makeclf': MAKECLF}
FAMILY_OF = {d: f for f, ds in FAMILIES.items() for d in ds}
DATASETS = [d for ds in FAMILIES.values() for d in ds]
NEW = NEW_REAL + GAUSSIAN + MAKECLF          # blind in the fifth registration

MODEL_GROUPS = {g: list(ms) for g, ms in sweep.MODEL_GROUPS.items()}
for _m in NEW_MODELS:
    MODEL_GROUPS.setdefault(NEW_GROUPS[_m], []).append(_m)
MODELS = [m for ms in MODEL_GROUPS.values() for m in ms]                  # 16
GROUP_OF = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}
GROUP_OF['Bayes Optimal'] = 'B quadratic'     # as sweep_gradient_truth.py
DIFFERENTIABLE = list(gradients.ANALYTIC_GRADIENTS)                       # 11
FIDELITY_MODELS = MODELS + [m for m in DIFFERENTIABLE if m not in MODELS]  # 17
CHECK_MODELS = FIDELITY_MODELS                                            # 17

assert len(DATASETS) == 71 and len(set(DATASETS)) == 71, len(DATASETS)
assert len(MODELS) == 16


def configure():
    '''point sweep.py's module lists at the full grid'''
    sweep.DATASETS = list(DATASETS)
    sweep.MODEL_GROUPS = {g: list(ms) for g, ms in MODEL_GROUPS.items()}
    sweep.MODELS = list(MODELS)
    sweep.GROUP_OF = dict(GROUP_OF)


def gaussian_params(name):
    '''(d, r, s) from a Gaussian-family name'''
    d, r, s = name.split()[1:]
    return int(d[1:]), int(r[1:]), int(s[1:])


def makeclf_params(name):
    '''(d, informative) from a make_classification name'''
    d, i = name.split()[1:]
    return int(d[1:]), int(i[1:])
