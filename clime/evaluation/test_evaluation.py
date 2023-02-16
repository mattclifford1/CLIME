# author: Matt Clifford
# email: matt.clifford@bristol.ac.uk
import inspect
from clime import evaluation

# def test_correct_args():
#     for eval in evaluation.AVAILABLE_FIDELITY_METRICS:
#         args_sig = inspect.signature(evaluation.AVAILABLE_FIDELITY_METRICS[eval])
#         args_list = list(args_sig.parameters.keys())
#         if args_list == ['explainer_generator', 'black_box_model', 'data', 'kwargs'] or args_list == ['explainer_generator', 'black_box_model', 'data', 'query_point', 'kwargs']:
#             assert True
#         else:
#             assert False
