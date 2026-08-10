'''
tests for the caching helpers and the plot axis scaling
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
import matplotlib
matplotlib.use('Agg')

from clime.utils import freezeargs
from clime.utils.plots import _get_ylims, _data_ylims, plot_line_graphs, plot_multiple_bar_dicts


def test_freezeargs_does_not_mutate_the_caller():
    '''
    regression test: freezing used to happen in place, handing the caller back
    their own dict with frozendict/tuple values
    '''
    @freezeargs
    def identity(d):
        return d

    opts = {'data params': {'class_samples': [20, 20]}}
    identity(opts)
    assert isinstance(opts['data params'], dict)
    assert type(opts['data params']) is dict, 'caller dict must stay a plain dict'
    assert isinstance(opts['data params']['class_samples'], list)


def test_freezeargs_result_is_hashable():
    @freezeargs
    def identity(d):
        return d

    hash(identity({'a': {'b': [1, 2]}}))


def test_bounded_metric_keeps_a_fixed_axis():
    '''fidelity must always be plotted on [0, 1] so runs stay comparable'''
    assert _get_ylims([0.94, 0.97], 'fidelity (local)') == [0, 1]


def test_unbounded_metric_is_scaled_to_the_data():
    '''
    regression test: Brier scores are ~0.02, and a hard coded [0, 1] axis
    rendered them as a flat line / single colour
    '''
    limits = _get_ylims([0.020, 0.025], 'Brier score (local)')
    assert limits[0] < 0.020 and limits[1] > 0.025
    assert limits[1] - limits[0] < 0.1, 'axis must be tight enough to see the variation'


def test_log_loss_axis_is_not_clipped():
    limits = _get_ylims([0.5, 7.3], 'log loss')
    assert limits[1] > 7.3, 'values above 1 must not be clipped away'


def test_data_ylims_ignores_nans():
    np.testing.assert_allclose(_data_ylims([np.nan, 0.0, 1.0], pad=0), [0.0, 1.0])


def test_data_ylims_constant_values():
    low, high = _data_ylims([0.5, 0.5])
    assert low < 0.5 < high


def test_plot_axis_defaults_do_not_leak_between_calls():
    '''
    regression test: ylims defaulted to a mutable [0, 1] that was expanded in
    place, so one plot with a large score rescaled every later plot in the session
    '''
    big = {0: {'a': {'scores': [0.0, 5.0]}}}
    small = {0: {'a': {'scores': [0.2, 0.3]}}}
    plot_line_graphs(big, ylabels=['log loss'])
    figure = plot_line_graphs(small, ylabels=['log loss'])
    limits = matplotlib.pyplot.gcf().axes[0].get_ylim()
    assert limits[1] < 1.0, f'axis leaked from the previous plot: {limits}'
    matplotlib.pyplot.close('all')


def test_bar_plot_axis_defaults_do_not_leak():
    plot_multiple_bar_dicts({0: {'a': {'avg': 5.0, 'std': 0.1}}}, ylabels=['log loss'])
    plot_multiple_bar_dicts({0: {'a': {'avg': 0.3, 'std': 0.1}}}, ylabels=['log loss'])
    limits = matplotlib.pyplot.gcf().axes[0].get_ylim()
    assert limits[1] < 1.0, f'axis leaked from the previous plot: {limits}'
    matplotlib.pyplot.close('all')


def test_sampling_is_reproducible_regardless_of_order():
    '''
    regression test (FINDINGS.md B10): neighbourhood sampling used to come from the
    global numpy stream, so results depended on how many draws preceded them - and under
    multiprocessing, on which worker took which query point
    '''
    import clime
    from clime.utils import rng_from_point
    q = np.array([0.3, -1.2, 4.0])
    first = rng_from_point(q, salt='a').normal(size=5)
    np.random.normal(size=54321)          # unrelated draws from the global stream
    second = rng_from_point(q, salt='a').normal(size=5)
    np.testing.assert_array_equal(first, second)


def test_different_salts_give_different_draws():
    '''
    the surrogate's training sample and the evaluation sample are both drawn around the
    query point - without different salts the surrogate would be scored on exactly the
    points it was fitted to
    '''
    from clime.utils import rng_from_point
    q = np.array([0.3, -1.2, 4.0])
    a = rng_from_point(q, salt='surrogate training sample').normal(size=5)
    b = rng_from_point(q, salt='local evaluation sample').normal(size=5)
    assert not np.allclose(a, b)


def test_different_query_points_give_different_draws():
    from clime.utils import rng_from_point
    a = rng_from_point(np.array([0.0, 0.0]), salt='s').normal(size=5)
    b = rng_from_point(np.array([0.0, 1.0]), salt='s').normal(size=5)
    assert not np.allclose(a, b)


def test_freezeargs_preserves_cache_control():
    '''
    functools.cache exposes cache_clear on the object, not in __dict__, so @wraps drops
    it. Without it there is no way to invalidate run_pipeline's cache when something
    outside the options dict changes (kernel width, random seed)
    '''
    from functools import cache
    from clime.utils import freezeargs

    calls = []

    @freezeargs
    @cache
    def counted(opts):
        calls.append(1)
        return len(calls)

    counted({'a': 1})
    counted({'a': 1})
    assert len(calls) == 1, 'second call should have been cached'
    counted.cache_clear()
    counted({'a': 1})
    assert len(calls) == 2, 'cache_clear did not invalidate'
