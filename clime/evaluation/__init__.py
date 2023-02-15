from .faithfulness import fidelity, local_fidelity, bal_fidelity

AVAILABLE_FIDELITY_METRICS = {
    'fidelity (class balanced)': bal_fidelity,
    'fidelity (local)': local_fidelity,
    'fidelity (normal)': fidelity,
}
