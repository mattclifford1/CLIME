# CLIME pipeline structure
Each part of the pipeline is split into its own folder.

| folder | stage(s) it provides | docs |
|---|---|---|
| [`data`](./data) | `dataset`, `dataset rebalancing` | [README](./data/README.md), [loaders](./data/loaders/readme.md) |
| [`models`](./models) | `model`, `model balancer` | [readme](./models/readme.md) |
| [`explainer`](./explainer) | `explainer` | [README](./explainer/README.md) |
| [`evaluation`](./evaluation) | `evaluation metric`, `evaluation points`, `evaluation data` | [README](./evaluation/README.md) |
| [`pipeline`](./pipeline) | assembles and runs the above | [README](./pipeline/README.md) |
| [`utils`](./utils) | plotting, notebook widgets, permutation helpers, caching | — |

## Add new methods
To add a new method to the pipeline, add the name and callable object to the respective dictionaries in the `__init__.py` files.

Make sure to follow the existing structure of methods are called (see the kwargs used in the [pipeline](./pipeline/make_pipeline.py)). Other helper base abstract classes exists too: eg. for [models](./models/base.py).

The registry key string is used directly as the plot label and as the notebook widget
entry, so renaming one changes every figure legend.

## Tests
Use pytest to collect and run all tests. The pipeline [test](./pipeline/test_pipeline.py) will run all possible configurations of the pipeline so is useful to make sure all features run.

Note that it asserts runs *complete*, not that they produce correct values.
