# CLIME pipeline structure
Each part of the pipeline is split into its own folder.

## Add new methods
To add a new method to the pipeline, add the name and callable object to the respective dictionaries in the `__init__.py` files.

Make sure to follow the existing structure of methods are called (see the kwargs used in the [pipeline](./pipeline/make_pipeline.py)). Other helper base abstract classes exists too: eg. for [models](./models/base.py).

## Tests
Use pytest to collect and run all tests. The pipeline [test](./pipeline/test_pipeline.py) will run all possible configurations of the pipeline so is useful to make sure all features run.
