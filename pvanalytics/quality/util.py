"""General-purpose quality control utility functions."""
import numpy as np


def check_limits(val, lower_bound=None, upper_bound=None,
                 inclusive_lower=False, inclusive_upper=False):
    """Check whether a value falls withing the given limits.

    At least one of `lower_bound` or `upper_bound` must be provided.

    Parameters
    ----------
    val : array_like
        Values to test.
    lower_bound : float, default None
        Lower limit.
    upper_bound : float, default None
        Upper limit.
    inclusive_lower : bool, default False
        Whether the lower bound is inclusive (`val` >= `lower_bound`).
    inclusive_upper : bool, default False
        Whether the upper bound is inclusive (`val` <= `upper_bound`).

    Returns
    -------
    array_like
        True for every value in `val` that is between `lower_bound`
        and `upper_bound`.

    Raises
    ------
    ValueError
        if `lower_bound` nor `upper_bound` is provided.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    if inclusive_lower:
        lb_op = np.greater_equal
    else:
        lb_op = np.greater
    if inclusive_upper:
        ub_op = np.less_equal
    else:
        ub_op = np.less

    if (lower_bound is not None) & (upper_bound is not None):
        return lb_op(val, lower_bound) & ub_op(val, upper_bound)
    elif lower_bound is not None:
        return lb_op(val, lower_bound)
    elif upper_bound is not None:
        return ub_op(val, upper_bound)
    else:
        raise ValueError('must provide either upper or lower bound')


def daily_min(series, minimum, inclusive=False):
    """Select only data on days where the daily minimum is less than minimum.

    Parameters
    ----------
    series : Series
        A Datetimeindexed series of floats.
    maximum : float
        The largest acceptable value for the daily minumum.
    inclusive : boolean, default False
        Whether values equal to `maximum` are considered too large.

    Returns
    -------
    Series
        True for every value that occurs on a day where the minimim
        value recorded on that day is less than `minimum`. False for
        all values on days where the daily minimum is >= minimum.

    Notes
    -----
    This function is derived from code in the pvfleets_qa_analysis
    project under the terms of the 3-clause BSD license. Copyright (c)
    2020 Alliance for Sustainable Energy, LLC.

    """
    operator = np.greater
    if inclusive:
        operator = np.greater_equal
    dailymin = series.resample('D').min()
    flags = operator(dailymin, minimum)
    return flags.reindex(index=series.index, method='ffill', fill_value=False)
