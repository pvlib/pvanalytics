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


def _daytime_powercurve(ac_power):
    # return the 99.5% quantile of daytime power data after removing
    # night time data.
    #
    # Copyright (c) 2020 Alliance for Sustainable Energy, LLC.
    power = ac_power.to_frame()
    power_column = power.columns[0]
    power['minutes'] = power.index.hour * 60 + power.index.minute

    power['positive_power'] = ac_power
    power.loc[power.positive_power >= 0, 'positive_power'] = 1
    powerfreq = power.groupby('minutes').positive_power.sum()
    daytimes = powerfreq[powerfreq > powerfreq.quantile(0.25)].index
    daytime_power = power[power.minutes.isin(daytimes)]

    return daytime_power.groupby('minutes')[power_column].quantile(0.995)


def _clipping_power(ac_power, clip_derivative=0.0035, freq=None):
    # Returns the clipping power threshold or None if no clipping is
    # identified in the data.
    #
    # Copyright (c) 2020 Alliance for Sustainable Energy, LLC.
    if not freq:
        freq = pd.Timedelta(pd.infer_freq(ac_power.index)).seconds * 60
    elif freq.isinstance(str):
        freq = pd.Timedelta(freq).seconds * 60

    # Use the derivative of the 99.5% quantile of daytime power at
    # each minute to identify clipping.
    powercurve = _daytime_powercurve(ac_power)
    normalized_power = powercurve / powercurve.max()
    power_derivative = (normalized_power.diff()
                        / normalized_power.index.to_series().diff()) * freq
    power_median = powercurve.median()

    oldcount = 0
    newcount = 0
    oldpowersum = 0.0
    newpowersum = 0.0
    for derivative, power in zip(power_derivative, powercurve):
        if ((np.abs(derivative) <= clip_derivative) and (power > power_median * 0.75)):
            newcount += 1
            newpowersum += power
        else:
            if newcount > oldcount:
                oldcount = newcount
                oldpowersum = newpowersum
            newcount = 0
            newpowersum = 0.0

    if (oldcount * freq) >= 60:
        return oldpowersum / oldcount

    return None


def threshold(ac_power, clip_derivative=0.0035, freq=None):
    """Detect clipping based on a maximum power threshold.

    A power threshold is calculated from the AC power data and any
    power above that threshold is assumed to be clipped. The threshold
    is computed based on the derivative of the 99.5% quantile of all
    power data grouped by time of day.

    Parameters
    ----------
    ac_power : Series
        Time series of AC Power.
    clip_derivative : float, default 0.0035
        Minimum derivative for clipping to be indicated. The default
        value has been derived empirically to prevent false positives
        for tracking PV systems.
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
    This function is based on code from the pvfleets_qa_analysis
    project. Copyright (c) 2020 Alliance for Sustainable Energy, LLC.

    """
    threshold = _clipping_power(
        ac_power,
        clip_derivative=clip_derivative,
        freq=freq
    )
    return ac_power >= threshold
