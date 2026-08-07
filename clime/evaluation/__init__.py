from .faithfulness import (fidelity, 
                           local_fidelity, 
                           bal_fidelity, 
                           local_and_bal_fidelity, 
                           rbig_kl, 
                           spearman, 
                           log_loss_score, 
                           local_log_loss_score,
                           kl_divergence,
                           local_kl_divergence,
                           Brier_score,
                           local_Brier_score,
                           query_probs_fidelity,
                           query_probs_local_fidelity)
from .key_points import get_key_points_score

AVAILABLE_EVALUATION_METRICS = {
    'Brier score (local)': local_Brier_score,
    'log loss (local)': local_log_loss_score,
    'fidelity (local)': local_fidelity,
    'fidelity (class balanced)': bal_fidelity,
    'KL divergence (local)': local_kl_divergence,
    'KL divergence': kl_divergence,
    # N.B. estimates mutual information, not a divergence - see faithfulness.rbig_kl
    'mutual information (RBIG)': rbig_kl,
    'spearman': spearman,
    'fidelity (local and balanced)': local_and_bal_fidelity,
    'fidelity (normal)': fidelity,
    'fidelity (query probs)': query_probs_fidelity,
    'fidelity (local query probs)': query_probs_local_fidelity,
    'log loss': log_loss_score,
    'Brier score': Brier_score,
}

# display range of each metric, used to scale plot axes
#   (min, max) -> a bounded metric, always plot on this fixed scale
#   None       -> unbounded (or very small range), derive limits from the data
# without this, scores such as Brier (~0.02) or log loss (unbounded) are
# squashed into - or clipped out of - a hard coded [0, 1] axis
METRIC_RANGES = {
    'Brier score (local)': None,
    'log loss (local)': None,
    'fidelity (local)': (0, 1),
    'fidelity (class balanced)': (0, 1),
    'KL divergence (local)': None,
    'KL divergence': None,
    'mutual information (RBIG)': None,
    'spearman': (-1, 1),
    'fidelity (local and balanced)': (0, 1),
    'fidelity (normal)': (0, 1),
    'fidelity (query probs)': (0, 1),
    'fidelity (local query probs)': (0, 1),
    'log loss': None,
    'Brier score': None,
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
