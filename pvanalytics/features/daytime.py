"""Functions for identifying daytime"""
import numpy as np
import pandas as pd


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


def _filter_midday_errors(night):
    # identify periods of time that appear to switch from night to day
    # (or day to night) on too short a time scale to be reasonable.
    # TODO make this adapable to the frequency of the series
    invalid = _run_lengths(night) <= 20
    # Need a numeric series so we can use Series.median()
    numeric_series = _to_numeric(night)
    smoothed = round(
        _rolling_by_minute(
            numeric_series,
            days=31,
            f=pd.core.window.RollingGroupby.median
        )
    )
    return (~invalid & night) | (invalid & _from_numeric(smoothed))


def _filter_and_normalize(series, outliers):
    # filter a series by removing outliers and clamping the minimum to
    # 0. Then normalize the series by the maximum deviation.
    series = series[~outliers]
    series.loc[series < 0] = 0
    return (series - series.min()) / (series.max() - series.min())


def diff(series, outliers=None, low_value_threshold=0.003,
         low_median_threshold=0.0015, low_diff_threshold=0.0015,
         clipping=None):
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

    """
    series_norm = _filter_and_normalize(
        series,
        outliers or pd.Series(False, index=series.index)
    )
    first_order_diff = series_norm.diff()
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
    # goes to 0 and stays there for several hours)
    night = _filter_midday_errors(night)
    # TODO optional validation against clearsky daylight hours
    # If a clipping mask was provided mark all clipped values as
    # daytime
    return (clipping or False) | (~night)
