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
    'Brier score (local)': local_Brier_score,
    'log loss (local)': local_log_loss_score,
    'fidelity (local)': local_fidelity,
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

AVAILABLE_EVALUATION_POINTS ={   # give the value to 'key_points' arg in get_key_points_score func
    'grid': 'grid',
    'between_class_means': 'between_means',
    'data_limits': 'data_edges',
    'class_means': 'means',
    'all_test_points': 'all_points',
}

AVAILABLE_EVALUATION_DATA = {   # give the value to 'test_points' arg in get_key_points_score func
    'sample locally': 'local',
    'test data': 'all',
}
