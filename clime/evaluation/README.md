# Evaluation Metrics
To add a new evalutation metric, add it to the `__init__.py` dictionary containing the store of all available methods - `AVAILABLE_EVALUATION_METRICS`.

### Format
When called the evaluation method is expected to take the key word arguments:
  - `expl`: scikit learn style model class (explainer)
  - `black_box_model`: scikit learn style model class
  - `data`: dictionary with 'X', 'y' etc. (see data section for more details)

Optional:
  - query_point: the query point used to train the local explainer (for use with local metrics)
Use `**kwargs` to peel off any arguments you dont want to use

Returns:
  - The evaluation needs to return the single evaluation score (float).

### Examples
See the given evaluation methods in `faithfulness.py` for an idea how to create your own.




# Running Evaluations
To add a new evaluation run type add it to the `__init__.py` dictionary containing the store of all available methods - `AVAILABLE_EVALUATION_POINTS`.

This uses the evaluation metric and calculates the score over a select number of query points (e.g. means of classes or the whole test dataset).

### Format
When called the evaluation runner is expected to take the key word arguments:
  - `metric`: one of the metrics from above (AVAILABLE_EVALUATION_METRICS)
  - `explainer_generator`: object to create an explanation (see below)
  - `black_box_model`: scikit learn style model class
  - `data`: dictionary with 'X', 'y' etc. (see data section for more details)
  - `run_parallel`: option to multiprocess evaluation


Returns:
  - Needs to return dict with keys 'avg' and 'std' of all the run. Optionally you can return 'eval_points' also which is a list of the query points used for plotting purposes.

### Generating explanations
To generate an explanation from the given generator use:
```
expl = explainer_generator(clf, data_dict, query_point_ind)
```

### Examples
See the given evaluation methods in `key_points.py` for an idea how to create your own.
