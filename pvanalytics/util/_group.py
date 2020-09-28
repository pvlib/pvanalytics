"""Functions for grouping data"""
import pandas as pd


def by_day(data):
    """Group data by day, preserving timezone in each group.

    This function can be used in place of ``pd.Series.resample('D')``
    when it is necessary to preserve the timezone in the index of each
    group (e.g. when the function being applied to the Resampler
    requires a localized time series).

    Parameters
    ----------
    data : Series
        DatetimeIndexed series.

    Returns
    -------
    GroupBy
        Data grouped by day, with a DatetimeIndex with daily frequency
        and the same timezone as `data`.

    """
    return data.groupby(
        pd.to_datetime(data.index.date).tz_localize(data.index.tz)
    )


def by_minute(data):
    """Group data by minute since midnight.

    Parameters
    ----------
    data : Series
        DatetimeIndexed series.

    Returns
    -------
    GroupBy
        Data grouped by minute since midnight (an Int64Index with
        values in the range 0 to 1440).

    """
    return data.groupby(data.index.minute + data.index.hour * 60)


def run_lengths(series):
    """
    Count the number of adjacent values that are equal.

    Paramters
    ---------
    series : Series

    Returns
    -------
    Series
        An integer for each value specifying the number of adjacent
        values that are equal.

    Examples
    --------
    >>> run_lengths(pd.Series([True, True, True]))
    0    3
    1    3
    2    3

    >>> run_lengths(
    ...     pd.Series([True, False, False, True, True, False]
    ... ))
    0    1
    1    2
    2    2
    3    2
    4    2
    5    1
    """
    runs = (series != series.shift(1)).cumsum()
    return runs.groupby(runs).transform('count')
