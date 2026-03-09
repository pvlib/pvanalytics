"""Functions for identifying and labeling outliers."""
import pandas as pd
import numpy as np
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


def zscore(data, zmax=1.5, nan_policy='raise'):
    """Identify outliers using the z-score.

    Points with z-score greater than `zmax` are considered as outliers.

    Parameters
    ----------
    data : Series
        A series of numeric values in which to find outliers.
    zmax : float
        Upper limit of the absolute values of the z-score.
    nan_policy : {'raise', 'omit'}, default 'raise'
        Define how to handle NaNs in the input series.
        If 'raise', a ValueError is raised when `data` contains NaNs.
        If 'omit', NaNs are ignored and False is returned at indices that
        contained NaN in `data`.

    Returns
    -------
    Series
        A series of booleans with True for each value that is an
        outlier.

    """
    nan_mask = pd.Series([False] * len(data),
                         index=data.index)

    if data.hasnans:
        if nan_policy == 'raise':
            raise ValueError("The input contains nan values.")
        elif nan_policy == 'omit':
            nan_mask = data.isna()
        else:
            raise ValueError(f"Unnexpected value ({nan_policy}) passed to "
                             "nan_policy. Expected 'raise' or 'omit'.")

    is_outlier = pd.Series(False, index=data.index)
    is_outlier.loc[~nan_mask] = abs(stats.zscore(data[~nan_mask])) > zmax
    return is_outlier


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


def compare_reference(actual, reference, comparison='difference',
                      method='zscore', **kwargs):
    """Identify outliers in `actual` based on deviations from `reference`.

    The `actual` and `reference` series are compared using the
    specified `comparison` type. The resulting deviations are then
    checked for outliers using `method`.

    Parameters
    ----------
    actual : Series
        The data in which to find outliers.
    reference : Series
        The reference data to compare against.
    comparison : {'difference', 'relative', 'absolute_difference'}, \
        default 'difference'
        How to compare `actual` and `reference`.
        If 'difference', then `actual - reference`.
        If 'relative', then `(actual - reference) / reference`.
        If 'absolute_difference', then `abs(actual - reference)`.
    method : {'zscore', 'tukey'}, default 'zscore'
        The method used to identify outliers in the deviations.
    **kwargs
        Passed to the outlier detection `method` (e.g. `zmax` for
        `zscore` or `k` for `tukey`).

    Returns
    -------
    Series
        A series of booleans with True for each value in `actual`
        that is an outlier.

    """
    if comparison == 'difference':
        deviations = actual - reference
    elif comparison == 'relative':
        deviations = (actual - reference) / reference
    elif comparison == 'absolute_difference':
        deviations = abs(actual - reference)
    else:
        raise ValueError(f"Invalid comparison '{comparison}'. "
                         "Expected 'difference', 'relative', or "
                         "'absolute_difference'.")

    if method == 'zscore':
        return zscore(deviations, **kwargs)
    elif method == 'tukey':
        return tukey(deviations, **kwargs)
    else:
        raise ValueError(f"Invalid method '{method}'. "
                         "Expected 'zscore' or 'tukey'.")


def quantile_threshold(x, y, quantile, formula='y ~ x'):
    """Estimate a threshold curve using quantile regression.

    This function uses quantile regression to find a threshold curve
    for the data in `y` as a function of `x`.

    Parameters
    ----------
    x : array_like
        The independent variable.
    y : Series
        The dependent variable.
    quantile : float
        The quantile to estimate (between 0 and 1).
    formula : str, default 'y ~ x'
        The model formula passed to `statsmodels.formula.api.quantreg`.
        The data frame passed to statsmodels will have columns named 'x'
        and 'y'.

    Returns
    -------
    Series
        The predicted threshold values at each point in `x`.

    """
    import statsmodels.formula.api as smf
    data = pd.DataFrame({'x': x, 'y': y})
    model = smf.quantreg(formula, data)
    res = model.fit(q=quantile)
    return res.predict(data)

