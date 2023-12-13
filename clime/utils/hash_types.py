from functools import wraps
from frozendict import frozendict

'''
adaptation of https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments
makes dict and lists hashable for use with caching
'''

def recursive_freeze(value):
    if isinstance(value, dict):
        for k, v in value.items():
            value[k] = recursive_freeze(v)
        return frozendict(value)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = recursive_freeze(v)
        return tuple(value)
    else:
        return value

# To unfreeze
def recursive_unfreeze(value):
    if isinstance(value, frozendict):
        value = dict(value)
        for k, v in value.items():
            value[k] = recursive_unfreeze(v)

    return value


def freezeargs(func):
    """
    Transform mutable dictionnary into immutable.
    Useful to be compatible with cache
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([recursive_freeze(arg) if isinstance(
            arg, dict) else arg for arg in args])
        kwargs = {k: recursive_freeze(v) if isinstance(
            v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped
