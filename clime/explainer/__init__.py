from .local import bLIMEy

def class_sample_weighted_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, class_weight_sampled=True, **kwargs)

def class_class_weighted_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, class_weight_data=True, **kwargs)

def data_rebalanced_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, rebalance_sampled_data=True, **kwargs)

AVAILABLE_EXPLAINERS = {
    'bLIMEy (normal)': bLIMEy,
    'bLIMEy (cost sensitive sampled)': class_sample_weighted_bLIMEy,
    'bLIMEy (cost sensitive class)': class_class_weighted_bLIMEy,
    'bLIMEy (rebalance data training)': data_rebalanced_bLIMEy,
}
