"""Functions for identifying daytime"""
import numpy as np
import pandas as pd
from pvanalytics import util
from pandas.tseries.frequencies import to_offset


def _rolling_by_minute(data, days, f):
    # apply `f` to a rolling window of length `days` at each minute of
    # the day.
    rolling = data.groupby(
        data.index.hour * 60 + data.index.minute
    ).rolling(
        min_periods=1,
        center=True,
        window=days
    )
    result = f(rolling).reset_index(0, drop=True)
    return result.sort_index()


def _run_lengths(series):
    # Count the number of equal values adjacent to each value.
    #
    # Examples
    # --------
    # >>> _run_lengths(pd.Series([True, True, True]))
    # 0    3
    # 1    3
    # 2    3
    #
    # >>> _run_lengths(
    # ...     pd.Series([True, False, False, True, True, False]
    # ... ))
    # 0    1
    # 1    2
    # 2    2
    # 3    2
    # 4    2
    # 5    1
    runs = (series != series.shift(1)).cumsum()
    return runs.groupby(runs).transform('count')


def _correct_if_invalid(series, invalid, correction_window):
    # For every time marked `invalid` replace the truth value in `series`
    # with the truth value at the same time in the majority of the
    # surrounding days. The number of surrounding days to examine is indicated
    # by `correction_window`.
    rolling_majority = _rolling_by_minute(
        series,
        days=correction_window,
        f=lambda x: x.sum() / x.count() > 0.5
    )
    return (~invalid & series) | (invalid & rolling_majority)


def _correct_midday_errors(night, minutes_per_value, hours_min,
                           correction_window):
    # identify periods of time that appear to switch from night to day
    # (or day to night) on too short a time scale to be reasonable.
    invalid = _run_lengths(night)*minutes_per_value <= hours_min*60
    return _correct_if_invalid(night, invalid, correction_window)


def _correct_edge_of_day_errors(night, minutes_per_value,
                                day_length_difference_max,
                                day_length_window, correction_window):
    # Identify day-time periods that are "too short" and replace
    # values with the majority truth-value for the same time in the
    # surrounding days.
    #
    # Because daylight savings shifts happen at night we cannot look
    # at night-length directly. Instead we look for too-short days and
    # flag the full day for correction. This may result in slightly
    # reduced accuracy for sunrise/sunset times on these days (even if
    # the day/night boundary at one end of the day - sunrise or sunset
    # - was correctly marked, it will be replaced with the rolling
    # median for that minute).
    day_periods = (~night).astype(int)
    day_length = (1 + day_periods.groupby(
        night.cumsum()).transform('sum')) * minutes_per_value
    # remove night time values so they don't interfere with the median
    # day length.
    day_length.loc[night] = np.nan
    day_length_median = day_length.rolling(
        window=str(day_length_window) + 'D'
    ).median()
    # flag days that are more than 30 minutes shorter than the median
    short_days = day_length < (day_length_median - day_length_difference_max)
    invalid = short_days.groupby(short_days.index.date).transform(
        lambda day: any(day)
    )
    return _correct_if_invalid(night, invalid, correction_window)


def _filter_and_normalize(series, outliers):
    # filter a series by removing outliers and clamping the minimum to
    # 0. Then normalize the series by the maximum deviation.
    if outliers is not None:
        series.loc[outliers] = np.nan
    series.loc[series < 0] = 0
    return (series - series.min()) / (series.max() - series.min())


def _freqstr_to_minutes(freqstr):
    return util.freq_to_timedelta(freqstr).seconds / 60


def power_or_irradiance(series, outliers=None,
                        low_value_threshold=0.003,
                        low_median_threshold=0.0015,
                        low_diff_threshold=0.0005, median_days=7,
                        clipping=None, freq=None,
                        correction_window=31, hours_min=5,
                        day_length_difference_max=30,
                        day_length_window=14):
    """Return True for values that are during the day.

    After removing outliers and normalizing the data, a time is
    classified as night when two of the following three criteria are
    satisfied:

    - near-zero value
    - near-zero first-order derivative
    - near-zero rolling median at the same time over the surrounding
      week (see `median_days`)

    Mid-day times where power goes near zero or
    stops changing may be incorrectly classified as night. To correct
    these errors, night or day periods with duration that is too long or
    too short are identified, and times in these periods are re-classified
    to have the majority value at the same time on preceding and
    following days (as set by `correction_window`).

    Finally any values that are True in `clipping` are marked as day.

    Parameters
    ----------
    series : Series
        Time series of power or irradiance.
    outliers : Series, optional
        Boolean time series with True for values in `series` that are
        outliers.
    low_value_threshold : float, default 0.003
        Maximum normalized power or irradiance value for a time to be
        considered night.
    low_median_threshold : float, default 0.0015
        Maximum rolling median of power or irradiance for a time to be
        considered night.
    low_diff_threshold : float, default 0.0005
        Maximum derivative of normalized power or irradiance for a time
        to be considered night.
    median_days : int, default 7
        Number of days to use to calculate the rolling median at each
        minute. [days]
    clipping : Series, optional
        True when clipping indicated. Any values where clipping is
        indicated are automatically considered 'daytime'.
    freq : str, optional
        A pandas freqstr specifying the expected timestamp spacing for
        the series. If None, the frequency will be inferred from the index.
    correction_window : int, default 31
        Number of adjacent days to examine when correcting
        day/night classification errors. [days]
    hours_min : float, default 5
        Minimum number of hours in a contiguous period of day or
        night. A day/night period shorter than `hours_min` is
        flagged for error correction. [hours]
    day_length_difference_max : float, default 30
        Days with length that is `day_length_difference_max` minutes less
        than the median length of surrounding days are flagged for
        corrections.
    day_length_window : int, default 14
        The length of the rolling window used for calculating the
        median length of the day when correcting errors in the morning
        or afternoon. [days]

    Returns
    -------
    Series
        Boolean time series with True for times that are during the
        day.

    Notes
    -----

    ``NA`` values are treated like zeros.

    References
    -------
    .. [1] Perry K., Meyers B., and Muller, M. "Survey of Time Shift Detection
       Algorithms for Measured PV Data", 2023 PV Reliability Workshop (PVRW).
    """
    series = series.fillna(value=0)
    series_norm = _filter_and_normalize(series, outliers).fillna(value=0)
    minutes_per_value = _freqstr_to_minutes(
        freq or pd.infer_freq(series.index)
    )
    first_order_diff = series_norm.diff() / minutes_per_value
    rolling_median = _rolling_by_minute(
        series_norm,
        days=median_days,
        f=pd.core.window.RollingGroupby.median
    )
    # Night-time if two of the following are satisfied:
    # - Near-zero value
    # - Near-zero first-order derivative
    # - Near-zero rolling median
    low_value = series_norm <= low_value_threshold
    low_median = rolling_median <= low_median_threshold
    low_diff = abs(first_order_diff) <= low_diff_threshold

    night = ((low_value & low_diff)
             | (low_value & low_median)
             | (low_diff & low_median))

    # Fix erroneous classifications (e.g. midday outages where power
    # goes to 0 and stays there for several hours, clipping classified
    # as night, and night-time periods that are too long)
    night_corrected_midday = _correct_midday_errors(
        night, minutes_per_value, hours_min, correction_window
    )
    night_corrected_clipping = ~((clipping or False)
                                 | (~night_corrected_midday))
    night_corrected_edges = _correct_edge_of_day_errors(
        night_corrected_clipping,
        minutes_per_value,
        day_length_difference_max,
        day_length_window,
        correction_window
    )
    return ~night_corrected_edges


def _get_sunrise_sunset_daily_series(daytime_mask, transform):
    # Get the sunset/sunrise series based on getting the first or last
    # 'day' value for each day in the time series
    series = daytime_mask.index[daytime_mask].to_series().groupby(
        daytime_mask[daytime_mask].index.date).transform(transform).reindex(
            daytime_mask.index)
    series = series.groupby(series.index.date).ffill().bfill()
    # Backfilling and front filling fills all NaN's, so we set cases not in
    # the right day to NaN
    series.loc[series.index.date != series.dt.date] = np.nan
    return series


def get_sunrise(daytime_mask, freq=None, data_alignment='L'):
    """
    Using the outputs of :py:func:`power_or_irradiance`, derive sunrise values
    for each day in the associated time series.

    This function assumes that each midnight-to-midnight period
    (according to the timezone of the input data) has one sunrise
    followed by one sunset. In cases where this is not satisfied
    (timezone of data is substantially different from the location's
    local time, locations near the poles, etc), or in the case of missing
    data, the returned sunrise and sunset times may be invalid.

    Parameters
    ----------
    daytime_mask  : Series
        Boolean series delineating night periods from day periods, where
        day is True and night is False.
    freq : str, optional
        A pandas freqstr specifying the expected timestamp spacing for
        the series. If None, the frequency will be inferred from the index of
        ``daytime_mask``.
    data_alignment  : str, default 'L'
        The data alignment of the series (left-aligned or right-aligned). Data
        alignment affects the value selected as sunrise. Options are 'L' (left-
        aligned), 'R' (right-aligned), or 'C' (center-aligned)

    Returns
    -------
    Series
        Series of daily sunrise times with the same index as ``daytime_mask``.

    References
    -------
    .. [1] Perry K., Meyers B., and Muller, M. "Survey of Time Shift Detection
       Algorithms for Measured PV Data", 2023 PV Reliability Workshop (PVRW).
    """
    # Get the first day period for each day
    sunrise_series = _get_sunrise_sunset_daily_series(daytime_mask, "first")
    # If there's no frequency value, infer it from the daytime_mask series
    if not freq:
        freq = pd.infer_freq(daytime_mask.index)
    # For left-aligned data, we want the first 'day' mask for
    # each day in the series; this will act as a proxy for sunrise.
    # Because of this, we will just return the sunrise_series with
    # no modifications
    if data_alignment == 'L':
        return sunrise_series
    # For center-aligned data, we want the mid-point between the last night
    # mask and the first day mask. To do this, we subtract freq / 2 from
    # each sunrise time in the sunrise_series.
    elif data_alignment == 'C':
        return (sunrise_series - (to_offset(freq) / 2))
    # For right-aligned data, get the last nighttime mask datetime
    # before the first 'day' mask in the series. To do this, we subtract freq
    # from each sunrise time in the sunrise_series.
    elif data_alignment == 'R':
        return (sunrise_series - to_offset(freq))
    else:
        # Throw an error if right,left, or center-alignment are not declared
        raise ValueError("No valid data alignment given. Please pass 'L'"
                         " for left-aligned data, 'R' for right-aligned data,"
                         " or 'C' for center-aligned data.")


def get_sunset(daytime_mask, freq=None, data_alignment='L'):
    """
    Using the outputs of :py:func:`power_or_irradiance`, derive sunset
    values for each day in the associated time series.

    This function assumes that each midnight-to-midnight period
    (according to the timezone of the input data) has one sunrise
    followed by one sunset. In cases where this is not satisfied
    (timezone of data is substantially different from the location's
     local time, locations near the poles, etc), or in the case of missing
    data, the returned sunrise and sunset times may be invalid.

    Parameters
    ----------
    daytime_mask  : Series
        Boolean series delineating night periods from day periods, where
        day is True and night is False.
    freq  : str, optional
        A pandas freqstr specifying the expected timestamp spacing for
        the series. If None, the frequency will be inferred from the index
        of ``daytime_mask``.
    data_alignment  : str, default 'L'
        The data alignment of the series (left-aligned or right-aligned). Data
        alignment affects the value selected as sunrise. Options are 'L' (left-
        aligned), 'R' (right-aligned), or 'C' (center-aligned)

    Returns
    -------
    Series
        Series of daily sunrise times with the same index as ``daytime_mask``.

    References
    -------
    .. [1] Perry K., Meyers B., and Muller, M. "Survey of Time Shift Detection
       Algorithms for Measured PV Data", 2023 PV Reliability Workshop (PVRW).
    """
    # Get the last day period for each day
    sunset_series = _get_sunrise_sunset_daily_series(daytime_mask, "last")
    # If there's no frequency value, infer it from the daytime_mask series
    if not freq:
        freq = pd.infer_freq(daytime_mask.index)
    #  For left-aligned data, sunset is the first nighttime period
    # after the day mask. To get this, we add freq to each sunset time in
    # the sunset time series.
    if data_alignment == 'L':
        return (sunset_series + to_offset(freq))
    # For center-aligned data, sunset is the midpoint between the last day
    # mask and the first nighttime mask. We calculate this by adding (freq / 2)
    # to each sunset time in the sunset_series.
    elif data_alignment == 'C':
        return (sunset_series + (to_offset(freq) / 2))
    # For right-aligned data, the last 'day' mask time stamp is sunset.
    elif data_alignment == 'R':
        return sunset_series
    else:
        # Throw an error if right, left, or center-alignment are not declared
        raise ValueError("No valid data alignment given. Please pass 'L'"
                         " for left-aligned data, 'R' for right-aligned data,"
                         " or 'C' for center-aligned data.")
