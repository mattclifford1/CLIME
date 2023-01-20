from clime import data, models, explainer, evaluation
from .make_pipeline import construct

AVAILABLE_MODULES = {
    'dataset': data.AVAILABLE_DATASETS,
    'dataset rebalancing': data.AVAILABLE_DATA_BALANCING,
    'model': models.AVAILABLE_MODELS,
    'model balancer': models.AVAILABLE_MODEL_BALANCING,
    'explainer': explainer.AVAILABLE_EXPLAINERS,
    'evaluation': evaluation.AVAILABLE_FIDELITY_METRICS,
}

AVAILABLE_MODULE_NAMES = list(AVAILABLE_MODULES.keys())
