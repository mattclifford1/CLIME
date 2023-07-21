from .base import base_model
from .linear import logistic, logistic_balanced_training
from .random_forest import random_forest, random_forest_balanced_training
# from .QDA import QDA
from .bayes_optimal import Guassian_class_conditional
from .svm import SVM, SVM_balanced_training
from .MLP import MLP_simple
from .balance import adjust_boundary, adjust_proba, base_balance
from .logit_regression import logit_ridge
from .logistic_regression import logistic_regression

AVAILABLE_MODELS = {
    'Random Forest': random_forest,
    'Random Forest balanced training': random_forest_balanced_training,
    'Logistic': logistic,
    'Logistic balanced training': logistic_balanced_training,
    'SVM': SVM,
    'SVM balanced training': SVM_balanced_training,
    # 'QDA': QDA,
    'Bayes Optimal': Guassian_class_conditional,
    'MLP': MLP_simple,
}

AVAILABLE_MODEL_BALANCING = {
    'none': base_balance,
    'boundary adjust': adjust_boundary,
    'probability adjust': adjust_proba,
}
