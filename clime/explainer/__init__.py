from .BLIMEY import bLIMEy
from .LIME import LIME_fatf
from .SHAP import kernal_SHAP

def sample_weighted_bLIMEy(*args, **kwargs):
    # add weights to the samples based on inverse 
    # occurance of minority class in the SAMPLED data
    return bLIMEy(*args, class_weight_sampled=True, **kwargs)

def sample_weighted_bLIMEy_logit(*args, **kwargs):
    # add weights to the samples based on inverse 
    # occurance of minority class in the SAMPLED data
    return bLIMEy(*args, class_weight_sampled=True, train_logits=True, **kwargs)

def bLIMEy_logit(*args, **kwargs):
    # add weights to the samples based on inverse 
    # occurance of minority class in the SAMPLED data
    return bLIMEy(*args, train_logits=True, **kwargs)

def sample_weighted_bLIMEy_probs(*args, **kwargs):
    # add weights to the samples based on inverse
    # occurance of minority class in the SAMPLED data
    # BUT* the opposite class is defined as points with 
    # probabilty lower than the query point
    return bLIMEy(*args, class_weight_sampled_probs=True, **kwargs)

def class_weighted_bLIMEy(*args, **kwargs):
    # add weights to the samples based on inverse 
    # occurance of minority class in the BLACK BOX TRAINING data
    return bLIMEy(*args, class_weight_data=True, **kwargs)

def data_rebalanced_bLIMEy(*args, **kwargs):
    return bLIMEy(*args, rebalance_sampled_data=True, **kwargs)

def dont_weight_locally_bLIMEy(*args, **kwargs):
    # dont add locality weightings when training surrogate model
    return bLIMEy(*args, weight_locally=False, **kwargs)

def just_class_weight_sampled_bLIMEy(*args, **kwargs):
    # dont add locality weightings when training surrogate model
    # BUT* do add BLACK BOX TRAINING data class imbalance costs
    return bLIMEy(*args, weight_locally=False, class_weight_sampled=True, **kwargs)


AVAILABLE_EXPLAINERS = {
    'bLIMEy (normal)': bLIMEy,
    'LIME (original)': LIME_fatf,
    'Kernel SHAP': kernal_SHAP,
    'bLIMEy (cost sensitive sampled)': sample_weighted_bLIMEy,
    'bLIMEy (cost sensitive sampled - probs)': sample_weighted_bLIMEy_probs,
    'bLIMEy (cost sensitive class)': class_weighted_bLIMEy,
    'bLIMEy (not local)': dont_weight_locally_bLIMEy,
    'bLIMEy (just cost sensitive sampled)': just_class_weight_sampled_bLIMEy,
    'bLIMEy (logit)': bLIMEy_logit,
    'bLIMEy (logit and sample weights)': sample_weighted_bLIMEy_logit,
    # 'bLIMEy (rebalance data training)': data_rebalanced_bLIMEy,
}
