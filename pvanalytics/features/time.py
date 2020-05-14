"""Features related to time."""
import numpy as np
import pandas as pd


def _minute_of_day(time):
    # return the minute of the day (counting from midnight).
    return time.hour * 60 + time.minute


def daytime_frequency(power_or_irradiance, threshold=0.8, minimum_days=60):
    """Identify daytime periods based on frequency of positive data.

    Data is aggregated by minute of the day and the mean number of
    positive values for each minute is calculated. Each minute with at
    least `threshold` times the mean number of positive values is
    considered day-time.

    Parameters
    ----------
    power_or_irradiance : Series
        DatetimeIndexed series of power or irradiance measurements.
    threshold : float, default 0.8
        Fraction of data that must have positive power or irradiance
        measurements for a time to be considered "day".
    minimum_days : int, default 60
        Minimum number of days with data. If `power_or_irradiance` has
        fewer days with positive data then a ValueError is raised.

    Returns
    -------
    Series
        A DatetimeIndexed series with True for each index that is
        during daylight hours.

    Raises
    ------
    ValueError
        if there are less than `minimum_days` of positive data.

    Notes
    -----
    This function is derived from the pvfleets_qa_analysis
    project. Copyright (c) 2020 Alliance for Susteainable Energy, LLC.

    """
    # remove all datapoints less than or equal to zero
    data = power_or_irradiance.copy()
    data[data <= 0] = np.nan
    minutes = pd.Series(_minute_of_day(data.index), index=data.index)
    # group by minute of day and compute frequency of positive values
    # at each minute
    value_frequency = data.groupby(minutes).count()
    if value_frequency.max() < minimum_days:
        raise ValueError("Too few days with data (got {}, minimum_days={})"
                         .format(value_frequency.max(), minimum_days))
    daylight_minutes = value_frequency[
        value_frequency > threshold * value_frequency.mean()
    ].index

    return minutes.isin(daylight_minutes)


def daytime_level(power_or_irradiance, threshold=0.2, quantile=0.95):
    """Identify daytime periods based on a minimum power/irradiance threshold.

    Power or irradiance data is aggregated by minute of day and times
    where the mean power is greater than `threshold` * max are marked
    as daytime, where max is the `quantile`-percent quantile of the
    data.

    Parameters
    ----------
    power_or_irradiance : Series
        DatetimeIndexed power or irradiance data.
    threshold : float, default 0.2
        Mean power at each minute of the day must be greater than or
        equal to `threshold` * max where max is the `quantile`-percent
        quantile of the data for the minute to be considered daytime.
    quantile : float, default 0.95
        Quantile to use as the upper bound of the data.

    Returns
    -------
    Series
        A series of booleans with True for timestamps when the sun is
        up.

    """
    max_power = power_or_irradiance.quantile(quantile)
    minutes = pd.Series(_minute_of_day(power_or_irradiance.index),
                        index=power_or_irradiance.index)
    mean_by_minute = power_or_irradiance.groupby(minutes).mean()
    daylight_minutes = mean_by_minute[
        mean_by_minute >= threshold * max_power
    ].index
    return minutes.isin(daylight_minutes)
