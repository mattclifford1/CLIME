from clime import data, models, explainer, evaluation
from .make_pipeline import construct, run_pipeline
from .multiple_runs import get_avg

AVAILABLE_MODULES = {
    'dataset': data.AVAILABLE_DATASETS,
    'dataset rebalancing': data.AVAILABLE_DATA_BALANCING,
    'model': models.AVAILABLE_MODELS,
    'model balancer': models.AVAILABLE_MODEL_BALANCING,
    'explainer': explainer.AVAILABLE_EXPLAINERS,
    'evaluation metric': evaluation.AVAILABLE_EVALUATION_METRICS,
    'evaluation run': evaluation.AVAILABLE_EVALUATION_POINTS,
}

AVAILABLE_MODULE_NAMES = list(AVAILABLE_MODULES.keys())
