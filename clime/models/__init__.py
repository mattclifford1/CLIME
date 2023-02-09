from .base import base_model
from .svm import SVM, SVM_balanced_training
from .balance import adjust_boundary, adjust_proba, none

AVAILABLE_MODELS = {
    'SVM': SVM,
    'SVM balanced training': SVM_balanced_training,
}

AVAILABLE_MODEL_BALANCING = {
    'none': none,
    'boundary adjust': adjust_boundary,
    'probability adjust': adjust_proba,
}
