from .faithfulness import fidelity, local_fidelity, bal_fidelity, local_and_bal_fidelity

AVAILABLE_FIDELITY_METRICS = {
    'fidelity (class balanced)': bal_fidelity,
    'fidelity (local)': local_fidelity,
    'fidelity (local and balanced)': local_and_bal_fidelity,
    'fidelity (normal)': fidelity,
}
