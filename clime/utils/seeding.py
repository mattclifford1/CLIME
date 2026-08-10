'''
deterministic per query point random number generators

Sampling used to come from numpy's global stream, which is seeded once at import. That
made every result depend on how many draws happened earlier - and under multiprocessing,
on which worker picked up which query point, so the same configuration run twice gave
different answers (see FINDINGS.md B10).

Seeding from the query point itself instead makes each explainer's neighbourhood
reproducible regardless of evaluation order, parallelism or what else ran first, while
still varying with clime.RANDOM_SEED so that repeated trials are possible.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import hashlib
import numpy as np
import clime


def rng_from_point(query_point, salt=''):
    '''
    a numpy Generator determined by (clime.RANDOM_SEED, query_point, salt)

    inputs:
        - query_point: the point being explained
        - salt: distinguishes independent draws around the *same* query point.
          N.B. this matters: the surrogate's training sample and the evaluation
          sample are both drawn around the query point, and without different salts
          they would be identical - the surrogate would be scored on its own
          training data.

    uses hashlib rather than hash(), whose output is salted per process
    '''
    point = np.ascontiguousarray(np.asarray(query_point, dtype=np.float64))
    digest = hashlib.sha256(point.tobytes() + str(salt).encode()).digest()[:8]
    offset = int.from_bytes(digest, 'big')
    return np.random.default_rng((int(clime.RANDOM_SEED) + offset) % (2**63))
