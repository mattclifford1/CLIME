'''
add this study's methods to clime's registries, without editing clime

CLAUDE.md's rule is that a new method is one entry in the relevant registry dict. These
entries belong to one study rather than to the project, so they are added at import time
from here instead of being written into `clime/`:

  - nothing in `clime/` changes, so the repo's own test suite and the other experiments
    are unaffected (they never import this package)
  - `clime.pipeline.AVAILABLE_MODULES` holds references to the same dict objects as
    `clime.models.AVAILABLE_MODELS` and friends, so updating those in place is enough for
    the pipeline, the notebook widgets and the permutation sweeper to see them

Importing `common` calls this, so every script here gets the same registry.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import clime
from . import degrade, weights

_done = False


def register():
    '''idempotent: safe to call from every module and every worker process'''
    global _done
    if _done:
        return
    clime.models.AVAILABLE_MODELS.update(degrade.MODELS)
    clime.data.AVAILABLE_DATA_BALANCING.update(degrade.DATA_BALANCING)
    clime.explainer.AVAILABLE_EXPLAINERS.update(weights.AVAILABLE)
    _done = True


def registered():
    return {'models': sorted(degrade.MODELS),
            'dataset rebalancing': sorted(degrade.DATA_BALANCING),
            'explainers': sorted(weights.AVAILABLE)}
