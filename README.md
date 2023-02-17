# CLIME --- Class-balanced Local Interpretale Model-agnostic Explainer
Investigation into how imbalanced data affects the surrogate explainer pipeline.

<!-- ![Alt text](https://github.com/mattclifford1/CLIME/blob/main/pics/overview.png?raw=true "Overview") -->

<!-- Try out quickly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattclifford1/CLIME/blob/main/experiments.ipynb) -->

The pipeline can be run from the jupyter-notebook [experiments](experiments.ipynb).

## Setup
Create python environment, e.g. via conda and install dependancies:
```
conda create -n clime python=3.9 -y
conda activate clime
pip install -e .
```

## Running the pipeline
For running experiments and playing around it is easiest to run the notebook via jupyter:
```
jupyter-notebook
```

## Dev tools
Test can be run with pytest (all configurations of the pipeline are run so this can take a few minutes when using a laptop):
```
pytest
```

## Package structure
Available methods for each part of the pipeline are declared in the `__init__.py` of each subfolder. Add all new methods developed there.

The pipeline is run from [make_pipeline.py](clime/pipeline/make_pipeline.py).
