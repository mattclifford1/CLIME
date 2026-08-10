from .base import base_model
from .linear import logistic, logistic_balanced_training
from .random_forest import random_forest, random_forest_balanced_training
from .bayes_optimal import Guassian_class_conditional
from .svm import SVM, SVM_balanced_training
from .MLP import MLP_simple
from .balance import adjust_boundary, adjust_proba, base_balance
from .logit_regression import logit_ridge
from .logistic_regression import logistic_regression
from .calibrated import random_forest_platt, random_forest_isotonic, gradient_boosting
from .log_odds_families import LDA, gaussian_naive_bayes, decision_tree, knn
from .QDA import QDA
from .constructed_log_odds import (polynomial_logistic, rbf_logistic,
                                  bagged_logistic, nearest_class_mean)

AVAILABLE_MODELS = {
    'Random Forest': random_forest,
    'Random Forest balanced training': random_forest_balanced_training,
    'Logistic': logistic,
    'Logistic balanced training': logistic_balanced_training,
    'SVM': SVM,
    'SVM balanced training': SVM_balanced_training,
    'QDA': QDA,
    'LDA': LDA,
    'Gaussian Naive Bayes': gaussian_naive_bayes,
    'Decision Tree': decision_tree,
    'k Nearest Neighbours': knn,
    'Bayes Optimal': Guassian_class_conditional,
    'MLP': MLP_simple,
    'Gradient Boosting': gradient_boosting,
    'Random Forest (Platt calibrated)': random_forest_platt,
    'Random Forest (isotonic calibrated)': random_forest_isotonic,
    # log-odds geometry imposed by construction rather than inferred from family
    'Nearest Class Mean': nearest_class_mean,
    'Polynomial Logistic (deg 2)': polynomial_logistic,
    'RBF Logistic (Nystroem)': rbf_logistic,
    'Bagged Logistic': bagged_logistic,
}

AVAILABLE_MODEL_BALANCING = {
    'none': base_balance,
    'boundary adjust': adjust_boundary,
    'probability adjust': adjust_proba,
}
