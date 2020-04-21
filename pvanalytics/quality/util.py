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
    Copyright (c) 2019 SolarArbiter. See the license in the
    docs/liscences.rst file at the top level of this distribution and
    at at `<https://pvanalytics.readthedocs.io/en/latest/
    licenses.html#solar-forecast-arbiter>`_.

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
