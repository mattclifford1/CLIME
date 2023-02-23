from .faithfulness import fidelity, local_fidelity, bal_fidelity, local_and_bal_fidelity
from .average_score import get_avg_score
from .key_points import get_key_points_means_score, get_key_points_points_between_means_score

AVAILABLE_EVALUATION_METRICS = {
    'fidelity (class balanced)': bal_fidelity,
    'fidelity (local)': local_fidelity,
    'fidelity (local and balanced)': local_and_bal_fidelity,
    'fidelity (normal)': fidelity,
}

AVAILABLE_EVALUATION_POINTS = {
    'all_test_points': get_avg_score(),
    'class_means': get_key_points_means_score(),
    'between_class_means': get_key_points_points_between_means_score(),
}
