import os
import warnings
warnings.filterwarnings("ignore")   # for tqdm notebook warning
import numpy as np
from .pipeline import make_pipeline

RANDOM_SEED = 42
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
np.random.seed(RANDOM_SEED)
