"""Functions for identifying clipping."""
import pandas as pd
import numpy as np
from pvanalytics import util


def _detect_levels(x, count=3, num_bins=100):
    """Identify plateau levels in data.

    Parameters
    ----------
    x : Series
        Data in which to find plateaus.
    count : int
        Number of pleataus to return.
    num_bins : int
        Number of bins to use in histogram that finds plateau levels.

    Returns
    -------
    list of tuples
        (left, right) values of the interval in x with a detected
        plateau, in decreasing order of count of x values in the
        interval. List length is given by `count`.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    hist, bin_edges = np.histogram(x, bins=num_bins, density=True)
    level_index = np.argsort(hist * -1)
    levels = [(bin_edges[i], bin_edges[i + 1]) for i in level_index[:count]]
    return levels, bin_edges


def _label_clipping(x, window, frac):
    # label points in a rolling window where the number of 1s is
    # greater than window*frac
    tmp = x.rolling(window).sum()
    y = (tmp >= window * frac) & x.astype(bool)
    return y


def levels(ac_power, window=4, fraction_in_window=0.75,
           rtol=5e-3, levels=2):
    """Label clipping in AC power data based on levels in the data.

    Parameters
    ----------
    ac_power : Series
        Time series of AC power measurements.
    window : int, default 4
        Number of data points in a window used to detect clipping.
    fraction_in_window : float, default 0.75
        Fraction of points which indicate clipping if AC power at each
        point is close to the plateau level.
    rtol : float, default 5e-3
        A point is close to a clipped level M if
        abs(ac_power - M) < rtol * max(ac_power)
    levels : int, default 2
        Number of clipped power levels to consider.

    Returns
    -------
    Series
        True when clipping is indicated.

    Notes
    -----
    Copyright (c) 2019 SolarArbiter. See the file
    LICENSES/SOLARFORECASTARBITER_LICENSE at the top level directory
    of this distribution and at `<https://github.com/pvlib/
    pvanalytics/blob/master/LICENSES/SOLARFORECASTARBITER_LICENSE>`_
    for more information.

    """
    power = ac_power.copy()
    power.dropna(inplace=True)
    num_bins = np.ceil(1.0 / rtol).astype(int)
    flags = pd.Series(index=power.index, data=False)
    power_plateaus, bins = _detect_levels(power, count=levels,
                                          num_bins=num_bins)
    for lower, upper in power_plateaus:
        temp = pd.Series(index=power.index, data=0.0)
        temp.loc[(power >= lower) & (power <= upper)] = 1.0
        flags = flags | _label_clipping(temp, window=window,
                                        frac=fraction_in_window)
    return flags.reindex_like(ac_power).fillna(False)


def _daytime_powercurve(ac_power, power_quantile):
    # return the `power_quantile` quantile of power at each minute of
    # the day.
    positive_power = ac_power[ac_power >= 0]
    return positive_power.groupby(
        positive_power.index.hour * 60 + positive_power.index.minute
    ).quantile(power_quantile)


def _clipped(power, slope, power_min, slope_max):
    # return a mask that is true where `power` is greater than
    # `power_min` and the absolute value of `slope` is less
    # than or equal to `slope_max`
    return (np.abs(slope) <= slope_max) & (power > power_min)


def _clipping_power(ac_power, slope_max, power_min,
                    power_quantile, freq=None):
    # Calculate a power threshold, above which the power is being
    # clipped.
    #
    # - The daytime power curve is calculated using
    #   _daytime_powercurve(). This function groups `ac_power` by
    #   minute of the day and returns the `power_quantile`-quantile of
    #   power for each minute.
    #
    # - Each timestamp in the daytime power curve that satisfies the
    #   clipping criteria[*] is flagged.
    #
    # - The clipping threshold is calculated as the mean power during the
    #   longest flagged period in the daytime power curve.
    #
    # [*] clipping criteria: a timestamp satisfies the clipping
    # criteria if the absolute value of the slope of the daytime power curve
    # is less than `slope_max` and the value of the daytime
    # power curve is greater than `power_min` times the median of the
    # daytime power curve.
    #
    # Based on the PVFleets QA Analysis project
    if not freq:
        freq = util.freq_to_timedelta(
            pd.infer_freq(ac_power.index)).seconds / 60
    elif isinstance(freq, str):
        freq = util.freq_to_timedelta(freq).seconds / 60

    # Use the slope of the 99.5% quantile of daytime power at
    # each minute to identify clipping.
    powercurve = _daytime_powercurve(
        ac_power, power_quantile
    )
    normalized_power = powercurve / powercurve.max()
    power_slope = (normalized_power.diff()
                   / normalized_power.index.to_series().diff()) * freq

    clipped_times = _clipped(powercurve, power_slope,
                             powercurve.median() * power_min,
                             slope_max)
    clipping_cumsum = (~clipped_times).cumsum()
    # get the value of the cumulative sum of the longest True span
    longest_clipped = clipping_cumsum.value_counts().idxmax()
    # select the longest span that satisfies the clipping criteria
    longest = powercurve[clipping_cumsum == longest_clipped]

    if longest.index.max() - longest.index.min() >= 60:
        # if the period of clipping is at least 60 minutes then we
        # have enough evidence to determine the clipping threshold.
        return longest.mean()
    return None


def threshold(ac_power, slope_max=0.0035, power_min=0.75,
              power_quantile=0.995, freq=None):
    """Detect clipping based on a maximum power threshold.

    This is a two-step process. First a clipping threshold is
    identified, then any values in `ac_power` greater than or equal to
    that threshold are flagged.

    The clipping threshold is determined by computing a 'daily power
    curve' which is the `power_quantile` quantile of all values in
    `ac_power` at each minute of the day. This gives a rough estimate
    of the maximum power produced at each minute of the day.

    The daily power curve is normalized by its maximum and the minutes
    of the day are identified where the normalized curve's slope is
    less than `slope_max`. If there is a continuous period of time
    spanning at least one hour where the slope is less than
    `slope_max` and the value of the normalized daily power curve is
    greater than `power_min` times the median of the normalized daily
    power curve then the data has clipping in it. If no sufficiently
    long period with both a low slope and high power exists then there
    is no clipping in the data.  The average of the daily power curve
    (not normalized) during the longest period that satisfies the
    criteria above is the clipping threshold.

    Parameters
    ----------
    ac_power : Series
        DatetimeIndexed series of AC power data.
    slope_max : float, default 0.0035
        Maximum absolute value of slope of AC power quantile for
        clipping to be indicated. The default value has been derived
        empirically to prevent false positives for tracking PV
        systems.
    power_min: float, default 0.75
        The power during periods with slope less than `slope_max` must
        be greater than `power_min` times the median normalized
        daytime power.
    power_quantile : float, default 0.995
        Quantile used to calculate the daily power curve.
    freq : string, default None
        A pandas string offset giving the frequency of data in
        `ac_power`. If None then the frequency is inferred from the
        series index.

    Returns
    -------
    Series
        True when `ac_power` is greater than or equal to the clipping
        threshold.

    Notes
    -----
    This function is based on the pvfleets_qa_analysis project.

    """
    threshold = _clipping_power(
        ac_power,
        slope_max,
        power_min,
        power_quantile,
        freq=freq
    )
    return ac_power >= threshold


def _freq_minutes(index, freq):
    """Return the frequency in minutes for `freq`. If `freq` is None
    then use the frequency inferred from `index`."""
    if freq is None:
        freq = pd.infer_freq(index)
    if freq is None:
        raise ValueError("cannot infer frequency")
    return util.freq_to_timedelta(freq).seconds / 60


def _apply_daily_mask(mask, data, transformation):
    """Apply `f` to the data selected by `mask` on each day.

    Parameters
    ----------
    mask : Series
        Boolean Series with same index as `data`
    data : Series
        Series with the data.
    transformation : str or function
        Any value that can be passed to ``Series.resample().transform()``.

    Returns
    -------
    Series
        Series with same index as `mask` and values assigned by applying
        transformation to data in ``data[mask]`` on each day.
    """
    data = data.copy()
    data[~mask] = np.nan
    return data.resample('D').transform(transformation)


def _threshold_mean(mask, data):
    """Return daily thresholds based on mean and standard deviation.

    Parameters
    ----------
    mask : Series
        Boolean series.
    data : Series
        Data with same index as `mask`.

    Returns
    -------
    minimum : Series
        `data` transformed to the mean of ``data[mask]`` minus 2 times
         the standard deviation of ``data[mask]`` on each day.
    maximum : Series
        `data` transformed to the mean of ``data[mask]`` plus 2 times
         the standard deviation of ``data[mask]`` on each day.
    """
    daily_mean = _apply_daily_mask(mask, data, 'mean')
    daily_std = _apply_daily_mask(mask, data, 'std')
    daily_clipped_max = daily_mean + 2 * daily_std
    daily_clipped_min = daily_mean - 2 * daily_std
    # In cases where the standard deviation is 0 (i.e. all the data is
    # identical) it is possible for the mean to be above the daily maximum
    # by a very small amount due to floating point rounding errors. To ensure
    # that rounding errors do not affect the final outcome we lower the daily
    # clipping minimum if it is greater than the maximum for that day and
    # raise the daily clipping maximum if it is less than the minimum for
    # that day.
    daily_min, daily_max = _threshold_minmax(mask, data)
    min_above_max = daily_clipped_min > daily_max
    max_below_min = daily_clipped_max < daily_min
    daily_clipped_min[min_above_max] = daily_max[min_above_max]
    daily_clipped_max[max_below_min] = daily_min[max_below_min]
    return daily_clipped_min, daily_clipped_max


def _threshold_minmax(mask, data):
    """Return daily thresholds based on min and max.

    Parameters
    ----------
    mask : Series
        Boolean series
    data : Series
        Data with same index as `mask`.

    Returns
    -------
    minimum : Series
        `data` transformed to have the minimum value from ``data[mask]``
        on each day.
    maximum : Series
        `data` transformed to have the maximum value from ``data[mask]``
        on each day.
    """
    daily_max = _apply_daily_mask(mask, data, 'max')
    daily_min = _apply_daily_mask(mask, data, 'min')
    return daily_min, daily_max


def _rolling_low_slope(ac_power, window, slope_max):
    """Return True for timestamps where the data has slope less
    than `slope_min` over a rolling window of length `window."""
    # Reverse the series to do a forward looking (left-labeled)
    # rolling max/min.
    rolling_max = ac_power[::-1].rolling(
        window=window).max().reindex_like(ac_power)
    rolling_min = ac_power[::-1].rolling(
        window=window).min().reindex_like(ac_power)
    # calculate an upper bound on the derivative
    derivative_max = ((rolling_max - rolling_min)
                      / ((rolling_max + rolling_min) / 2) * 100)
    clipped = derivative_max < slope_max
    clipped_windows = clipped.copy()
    # flag all points in a window that has clipping
    for i in range(0, window):
        clipped_windows |= clipped.shift(i)
    return clipped_windows


def geometric(ac_power, window=None, slope_max=0.2, freq=None,
              tracking=False):
    """Identify clipping based on a the shape of the `ac_power`
    curve on each day.

    Each day is checked for periods where the slope of `ac_power`
    is small. The power values in these periods are used to calculate
    a minimum and a maximum clipped power level for that day. Any
    power values that are within this range are flagged as
    clipped. The methodology for computing the thresholds varies
    depending on the frequency of `ac_power`. For high frequency
    data (less than 10 minute timestamp spacing) the minimum
    clipped power is the mean of the low-slope period(s) on that
    day minus 2 times the standard deviation in the same period(s).
    For lower frequency data the absolute minimum and maximum of
    the low slope period(s) on each day are used.

    If the frequency of `ac_power` is less than ten minutes, then
    `ac_power` is down-sampled to 15 minutes and the mean value in
    each 15-minute period is used to reduce noise inherent in
    high frequency data.

    Parameters
    ----------
    ac_power : Series
        AC power data.
    window : int, optional
        Size of the rolling window used to identify low-slope
        periods. If not specified and `tracking` is False then
        `window=3` is used. If not specified and `tracking` is
        True then `window=5` is used.
    slope_max : float, default 0.2
        Maximum difference in maximum and minimum power for a
        window to be flagged as clipped. Units are percent of
        average power in the interval.
    freq : str, optional
        Frequency of `ac_power`. If not specified then
        :py:func:`pandas.infer_freq` is used.
    tracking : bool, default False
        If True then a larger default `window` is used. If `window`
        is specified then `tracking` has no effect.

    Returns
    -------
    Series
        Boolean Series with True for values that appear to be clipped.

    Raises
    ------
    ValueError
        If the index of `ac_power` is not sorted.

    Notes
    -----
    Based on code from the PVFleets QA project.
    """
    if not ac_power.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing.")
    ac_power_original = ac_power.copy()
    ac_power = ac_power_original
    try:
        freq_minutes = _freq_minutes(ac_power.index, freq)
    except ValueError:
        raise ValueError("Cannot infer frequency of `ac_power`. "
                         "Please resample or pass `freq`.")
    if freq_minutes < 10:
        ac_power = ac_power.resample('15T').mean()
    if window is None and tracking and freq_minutes < 30:
        window = 5
    else:
        window = window or 3
    # remove low power times to eliminate night.
    daily_min = ac_power.resample('D').transform('max') * 0.1
    ac_power.loc[ac_power < daily_min] = np.nan
    clipped = _rolling_low_slope(ac_power, window, slope_max)
    if not ac_power.index.equals(ac_power_original.index):
        # data was down-sampled.
        daily_clipped_min, daily_clipped_max = _threshold_mean(
            clipped.reindex_like(ac_power_original, method='ffill'),
            ac_power_original
        )
    else:
        daily_clipped_min, daily_clipped_max = _threshold_minmax(
            clipped, ac_power_original
        )
    return ((ac_power_original >= daily_clipped_min)
            & (ac_power_original <= daily_clipped_max))
