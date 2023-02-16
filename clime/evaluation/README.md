# Evaluation
To add a new evalutation method, add it to the `__init__.py` dictionary containing the store of all available methods.

## Format
When called the evaluation method is expected to take the key word arguments:
  - `explainer_generator`: object to create an explanation (see below)
  - `black_box_model`: scikit learn style model class
  - `data`: dictionary with 'X', 'y' etc. (see data section for more details)
  - `run_parallel`: option to multiprocess evaluation

Use `**kwargs` to peel off any arguments you dont want to use

## Generating explanations
To generate an explanation from the given generator use:
```
expl = explainer_generator(clf, data_dict, query_point_ind)
```

## Examples
See the given evaluation methods in `faithfulness.py` for an idea how to create your own.
