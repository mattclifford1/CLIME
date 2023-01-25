from .base import base_model
from .svm import *
from .balance import *

AVAILABLE_MODELS = {
    'SVM': SVM,
    'SVM balanced training': SVM_balanced_training,
}

AVAILABLE_MODEL_BALANCING = {
    'boundary adjust': adjust_boundary,
    'probability adjust': adjust_proba,
    'none': none,
}
