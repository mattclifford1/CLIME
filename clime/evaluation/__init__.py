from .faithfulness import fidelity, local_fidelity, bal_fidelity, local_and_bal_fidelity
from .average_score import get_avg_score
from .key_points import get_key_points_score

AVAILABLE_FIDELITY_METRICS = {
    'fidelity (class balanced)': get_avg_score(bal_fidelity),
    'fidelity (local)': get_avg_score(local_fidelity),
    'fidelity (local and balanced)': get_avg_score(local_and_bal_fidelity),
    'fidelity (normal)': get_avg_score(fidelity),
    'fidelity key (normal)': get_key_points_score(fidelity),
}
