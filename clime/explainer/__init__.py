from .BLIMEY import bLIMEy
from .LIME import LIME_fatf

def sample_weighted_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, class_weight_sampled=True, **kwargs)

def class_weighted_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, class_weight_data=True, **kwargs)

def data_rebalanced_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, rebalance_sampled_data=True, **kwargs)

def weight_locally_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, weight_locally=False, **kwargs)

def just_class_weight_sampled_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, weight_locally=False, class_weight_sampled=True, **kwargs)

AVAILABLE_EXPLAINERS = {
    'bLIMEy (normal)': bLIMEy,
    'bLIMEy (cost sensitive sampled)': sample_weighted_bLIMEy,
    'bLIMEy (cost sensitive class)': class_weighted_bLIMEy,
    'bLIMEy (not local)': weight_locally_bLIMEy,
    'bLIMEy (just cost sensitive sampled)': just_class_weight_sampled_bLIMEy,
    # 'bLIMEy (rebalance data training)': data_rebalanced_bLIMEy,
    'LIME (original)': LIME_fatf,
}
