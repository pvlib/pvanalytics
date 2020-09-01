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
