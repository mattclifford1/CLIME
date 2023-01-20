from .local import *

def class_weighted_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, class_weight=True, **kwargs)

def data_rebalanced_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, rebalance_sampled_data=True, **kwargs)

AVAILABLE_EXPLAINERS = {
    'normal': bLIMEy,
    'cost sensitive training': class_weighted_bLIMEy,
    'rebalance data training': data_rebalanced_bLIMEy,
}
