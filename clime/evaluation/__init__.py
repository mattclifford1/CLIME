from .faithfulness import fidelity, local_fidelity, bal_fidelity

AVAILABLE_FIDELITY_METRICS = {
    'fidelity (normal)': fidelity,
    'fidelity (local)': local_fidelity,
    'fidelity (class balanced)': bal_fidelity,
}
