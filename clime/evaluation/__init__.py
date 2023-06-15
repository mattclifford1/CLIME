from .faithfulness import (fidelity, 
                           local_fidelity, 
                           bal_fidelity, 
                           local_and_bal_fidelity, 
                           rbig_kl, 
                           spearman, 
                           log_loss_score, 
                           local_log_loss_score,
                           Brier_score,
                           local_Brier_score,
                           query_probs_fidelity,
                           query_probs_local_fidelity)
from .average_score import get_avg_score
from .key_points import get_key_points_score

AVAILABLE_EVALUATION_METRICS = {
    'fidelity (local)': local_fidelity,
    'log loss (local)': local_log_loss_score,
    'Brier score (local)': local_Brier_score,
    'fidelity (class balanced)': bal_fidelity,
    'KL': rbig_kl,
    'spearman': spearman,
    'fidelity (local and balanced)': local_and_bal_fidelity,
    'fidelity (normal)': fidelity,
    'fidelity (query probs)': query_probs_fidelity,
    'fidelity (local query probs)': query_probs_fidelity,
    'log loss': log_loss_score,
    'Brier score': Brier_score,
}

AVAILABLE_EVALUATION_POINTS = {
    'between_class_means': get_key_points_score(key_points='between_means', test_points='all'),
    'data_limits': get_key_points_score(key_points='data_edges', test_points='all'),
    'class_means': get_key_points_score(key_points='means', test_points='all'),
    'all_test_points': get_key_points_score(key_points='all_points', test_points='all'),
    'between_class_means_local': get_key_points_score(key_points='between_means', test_points='local'),
    'class_means_local': get_key_points_score(key_points='means', test_points='local'),
    'all_test_points_local': get_key_points_score(key_points='all_points', test_points='local'),
}
