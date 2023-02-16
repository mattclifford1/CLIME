from .faithfulness import get_avg_score, fidelity, local_fidelity, bal_fidelity, local_and_bal_fidelity

AVAILABLE_FIDELITY_METRICS = {
    'fidelity (class balanced)': get_avg_score(bal_fidelity),
    'fidelity (local)': get_avg_score(local_fidelity),
    'fidelity (local and balanced)': get_avg_score(local_and_bal_fidelity),
    'fidelity (normal)': get_avg_score(fidelity),
}
