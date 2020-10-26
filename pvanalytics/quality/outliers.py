"""Functions for identifying and labeling outliers."""
import pandas as pd
from scipy import stats
from statsmodels import robust


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


def hampel(data, window=5, max_deviation=3.0, scale=None):
    r"""Identify outliers by the Hampel identifier.

    The Hampel identifier is computed according to [1]_.

    Parameters
    ----------
    data : Series
        The data in which to find outliers.
    window : int or offset, default 5
        The size of the rolling window used to compute the Hampel
        identifier.
    max_deviation : float, default 3.0
        Any value with a Hampel identifier > `max_deviation` standard
        deviations from the median is considered an outlier.
    scale : float, optional
        Scale factor used to estimate the standard deviation as
        :math:`MAD / scale`. If `scale=None` (default), then the scale
        factor is taken to be ``scipy.stats.norm.ppf(3/4.)`` (approx. 0.6745),
        and :math:`MAD / scale` approximates the standard deviation
        of Gaussian distributed data.

    Returns
    -------
    Series
        True for each value that is an outlier according to its Hampel
        identifier.

    References
    ----------
    .. [1] Pearson, R.K., Neuvo, Y., Astola, J. et al. Generalized
       Hampel Filters. EURASIP J. Adv. Signal Process. 2016, 87
       (2016). https://doi.org/10.1186/s13634-016-0383-6

    """
    median = data.rolling(window=window, center=True).median()
    deviation = abs(data - median)
    kwargs = {}
    if scale is not None:
        kwargs = {'c': scale}
    mad = data.rolling(window=window, center=True).apply(
        robust.scale.mad,
        kwargs=kwargs
    )
    return deviation > max_deviation * mad
