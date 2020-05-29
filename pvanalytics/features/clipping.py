"""Functions for identifying clipping."""
import pandas as pd
import numpy as np


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
    num_bins = np.ceil(1.0 / rtol).astype(int)
    flags = pd.Series(index=ac_power.index, data=False)
    power_plateaus, bins = _detect_levels(ac_power, count=levels,
                                          num_bins=num_bins)
    for lower, upper in power_plateaus:
        temp = pd.Series(index=ac_power.index, data=0.0)
        temp.loc[(ac_power >= lower) & (ac_power <= upper)] = 1.0
        flags = flags | _label_clipping(temp, window=window,
                                        frac=fraction_in_window)
    return flags


def _daytime_powercurve(ac_power, power_quantile, frequency_quantile):
    # return the `power_quantile` quantile of power at each minute of
    # the day after removing night time, early morning, and late
    # evening data.
    minutes = ac_power.index.hour * 60 + ac_power.index.minute
    positive_power = ac_power >= 0
    powerfreq = positive_power.groupby(minutes).sum()
    min_daytime_freq = powerfreq.quantile(frequency_quantile)
    daytimes = powerfreq[powerfreq >= min_daytime_freq].index
    daytime_power = ac_power[minutes.isin(daytimes)]

    return daytime_power.groupby(
        daytime_power.index.hour * 60 + daytime_power.index.minute
    ).quantile(power_quantile)


def _clipped(power, slope, power_min, slope_max):
    # return a mask that is true where `power` is greater than
    # `power_min` and the absolute value of `slope` is less
    # than or equal to `slope_max`
    return (np.abs(slope) <= slope_max) & (power > power_min)


def _clipping_power(ac_power, slope_max, power_min,
                    power_quantile, frequency_quantile,
                    freq=None):
    # Calculate a power threshold, above which the power is being
    # clipped.
    #
    # - The daytime power curve is calculated using
    #   _daytime_powercurve(). This function groups `ac_power` by
    #   minute of the day and returns the `power_quantile`-quantile of
    #   power for each minute, excluding night time, early morning,
    #   and late afternoon/evening (the amount of data that is
    #   excluded is controlled by `frequency_quantile`).
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
        freq = pd.Timedelta(pd.infer_freq(ac_power.index)).seconds / 60
    elif isinstance(freq, str):
        freq = pd.Timedelta(freq).seconds / 60

    # Use the slope of the 99.5% quantile of daytime power at
    # each minute to identify clipping.
    powercurve = _daytime_powercurve(
        ac_power, power_quantile, frequency_quantile
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

    if longest.index[-1] - longest.index[0] >= 60:
        # if the period of clipping is at least 60 minutes then we
        # have enough evidence to determine the clipping threshold.
        return longest.mean()
    return None


def threshold(ac_power, slope_max=0.0035, power_min=0.75,
              power_quantile=0.995, frequency_quantile=0.25,
              freq=None):
    """Detect clipping based on a maximum power threshold.

    A clipping threshold is calculated from the AC power data and any
    power greater than the threshold is flagged as clipped.

    To calculate the clipping threshold, `ac_power` is aggregated at
    each minute of the day. Low power data is removed to eliminate
    night-time periods and the 99.5% quantile is computed at each
    minute. If the absolute value of the slope of the 99.5%
    quantile is less than `slope_max` for
    a continuous period of at least one hour then clipping is
    indicated. The mean power for that period is used as the
    threshold. If there are multiple periods with a slope less
    than `slope_max` then the longest period is used to compute
    the threshold.

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
        be greater than `power_min` time the median daytime power.
    power_quantile : float, default 0.995
        quantile of AC power to consider for determining clipping.
    frequency_quantile : float, default 0.25
        After taking the count of positive values in `ac_power` at
        each minute of the day, any minute with a count less than the
        `frequency_quantile`-quantile of all counts is excluded from
        the calculation of the clipping threshold.
    freq : string, default None
        A pandas string offset giving the frequency of data in
        `ac_power`. If None then the frequency is infered from the
        series index.

    Returns
    -------
    Series
        True when clipping is indicated.

    Notes
    -----
    This function is based on the pvfleets_qa_analysis project.

    """
    threshold = _clipping_power(
        ac_power,
        slope_max,
        power_min,
        power_quantile,
        frequency_quantile,
        freq=freq
    )
    return ac_power >= threshold
