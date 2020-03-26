"""Functions for identifying and labeling outliers."""
import pandas as pd
from scipy import stats


def tukey(data, k=1.5):
    r"""Identify outliers based on the interquartile range.

    A value `x` is considered an outlier if it does *not* satisfy the
    following condition

    .. math::
        Q_1 - k(Q_3 - Q_1) \le x \le Q_3 + k(Q_3 - Q_1)

    where :math:`Q_1` is the value of the first quartile and
    :math:`Q_3` is the value of the third quartile.

    Parameters
    ----------
    data : Series
        The data in which to find outliers.
    k : float, default 1.5
        Multiplier of the interquartile range. A larger value will be more
        permissive of values that are far from the median.

    Returns
    -------
    Series
        A series of booleans with True for each value that is an
        outlier.

    """
    first_quartile = data.quantile(0.25)
    third_quartile = data.quantile(0.75)
    iqr = third_quartile - first_quartile
    return ((data < (first_quartile - k*iqr))
            | (data > (third_quartile + k*iqr)))


def zscore(data, zmax=1.5):
    """Identify outliers using the z-score.

    Points with z-score greater than `zmax` are considered as outliers.
    the value is considered an outlier.

    Parameters
    ----------
    data : Series
        A series of numeric values in which to find outliers.
    zmax : float
        Upper limit of the absolute values of the z-score.

    Returns
    -------
    Series
        A series of booleans with True for each value that is an
        outlier.

    """
    return pd.Series((abs(stats.zscore(data)) > zmax), index=data.index)
