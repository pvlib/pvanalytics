"""Functions for identifying daytime"""
import numpy as np
import pandas as pd
from pandas.tseries import frequencies


def _rolling_by_minute(data, days, f, sort=True):
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
    if sort:
        return result.sort_index()
    return result


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


def _to_numeric(series):
    # Convert a Boolean series to a numeric series
    #
    # False -> 1, True -> 2, NA -> 3
    numeric_series = pd.Series(index=series.index, dtype='float64')
    numeric_series.loc[~series] = 1
    numeric_series.loc[series] = 2
    numeric_series.loc[series.isna()] = 3
    return numeric_series


def _from_numeric(series):
    # Inverse of :py:func:`_to_numeric()`
    boolean_series = series == 2
    boolean_series.loc[series == 3] = np.nan
    return boolean_series.astype('bool')


def _smooth_if_invalid(series, invalid):
    # replace values in `series` marked as True in `invalid` with the
    # 31-day rolling median at each minute.
    smoothed = round(
        _rolling_by_minute(
            _to_numeric(series),
            days=31,
            f=pd.core.window.RollingGroupby.median
        )
    )
    return (~invalid & series) | (invalid & _from_numeric(smoothed))


def _filter_midday_errors(night, minutes_per_value):
    # identify periods of time that appear to switch from night to day
    # (or day to night) on too short a time scale to be reasonable.
    invalid = _run_lengths(night)*minutes_per_value <= 5*60  # 5 hours
    return _smooth_if_invalid(night, invalid)


def _filter_edge_of_day_errors(night, minutes_per_value):
    # identify night-time periods that are "too long" and replace
    # values with the 31-day rolling median value for that minute.
    #
    # Because daylight savings shifts happen at night we cannot look
    # at night-length directly. Instead we look for too-short days and
    # flag the full day for substitution. This may result in slightly
    # reduced accuracy for sunrise/sunset times on these days (even if
    # one end of the day - sunrise or sunset - was correctly flagged,
    # it will be replaced with the rolling median for that minute).
    day_length = night.groupby(night.cumsum()).transform(
        lambda x: len(x) * minutes_per_value
    )
    # remove night time values so they don't interfer with the mediand
    # day length.
    day_length.loc[night] = np.nan
    day_length_median = day_length.rolling(window='14D').median()
    # flag days that are more than 30 minutes shorter than the median
    short_days = day_length < (day_length_median - 30)
    invalid = short_days.groupby(short_days.index.date).transform(
        lambda day: any(day)
    )
    return _smooth_if_invalid(night, invalid)


def _filter_and_normalize(series, outliers):
    # filter a series by removing outliers and clamping the minimum to
    # 0. Then normalize the series by the maximum deviation.
    series = series[~outliers]
    series.loc[series < 0] = 0
    return (series - series.min()) / (series.max() - series.min())


def _freqstr_to_minutes(freqstr):
    return pd.to_timedelta(
        frequencies.to_offset(freqstr)
    ).seconds / 60


def diff(series, outliers=None, low_value_threshold=0.003,
         low_median_threshold=0.0015, low_diff_threshold=0.0005,
         clipping=None, freq=None):
    """Return True for values that are during the day.

    Parameters
    ----------
    series : Series
        Time series of power or irradiance.
    outliers : Series, optional
        Boolean time series with True for values in `series` that are
        outliers.
    low_value_threshold : float, default 0.003
        Maximum normalized value for a time to be considered night.
    low_median_threshold : float, default 0.0015
        Minimum rolling median for a time to be considered night.
    clipping : Series, optional
        True when clipping indicated. Any values where clipping is
        indicated are automatically considered 'daytime'.

    Returns
    -------
    Series
        Boolean time series with True for times that are during the
        day.

    Notes
    -----

    ``NA`` values are treated like zeros.

    Derived from the PVFleets QA Analysis project.
    """
    series = series.fillna(value=0)
    series_norm = _filter_and_normalize(
        series,
        outliers or pd.Series(False, index=series.index)
    )
    minutes_per_value = _freqstr_to_minutes(
        freq or pd.infer_freq(series.index)
    )
    first_order_diff = series_norm.diff() / minutes_per_value
    rolling_median = _rolling_by_minute(
        series_norm,
        days=7,
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
    night_corrected_midday = _filter_midday_errors(night, minutes_per_value)
    night_corrected_clipping = ~((clipping or False)
                                 | (~night_corrected_midday))
    night_corrected_edges = _filter_edge_of_day_errors(
        night_corrected_clipping, minutes_per_value
    )
    return ~night_corrected_edges
