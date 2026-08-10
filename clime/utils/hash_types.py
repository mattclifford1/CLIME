from functools import wraps
from frozendict import frozendict

'''
adaptation of https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments
makes dict and lists hashable for use with caching
'''

def recursive_freeze(value):
    '''
    N.B. builds new containers rather than freezing in place - freezing in place
    would hand the caller's own dict back to them with frozendict/tuple values
    '''
    if isinstance(value, dict):
        return frozendict({k: recursive_freeze(v) for k, v in value.items()})
    elif isinstance(value, list):
        return tuple(recursive_freeze(v) for v in value)
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

    # functools.cache exposes these on the object rather than in __dict__, so @wraps does
    # not carry them across. Without this there is no way to invalidate the cache of a
    # function that is both frozen and cached - needed when something outside the options
    # dict changes, e.g. the locality kernel width or the random seed
    wrapped.cache_clear = getattr(func, 'cache_clear', lambda: None)
    wrapped.cache_info = getattr(func, 'cache_info', lambda: None)
    return wrapped
