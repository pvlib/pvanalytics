"""Features related to time."""
import numpy as np


def _minute_of_day(time):
    # return the minute of the day (counting from midnight).
    return time.hour * 60 + time.minute


def daytime(power_or_irradiance, threshold=0.8, minimum_days=60):
    """Get the times where the sun is up.

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
        if there are less than 60 days of actual data.

    Notes
    -----
    This function is derived from the pvfleets_qa_analysis
    project. Copyright (c) 2020 Alliance for Susteainable Energy, LLC.

    """
    # remove all datapoints less than or equal to zero
    data = power_or_irradiance.copy()
    data[data <= 0] = np.nan
    data = data.to_frame()
    data['minute'] = _minute_of_day(data.index)
    # group by minute of day and compute frequency of positive values
    # at each minute
    value_frequency = data.groupby('minute').count()

    if max(value_frequency) <= minimum_days:
        raise ValueError("Too few days with data (got {}, minimum_days={})"
                         .format(max(value_frequency), minimum_days))
    # TODO Ask Matt about the "adaptive" step where the threshold is
    #      increased from 0.8 to 0.9 if not enough data has been
    #      excluded.
    daylight_minutes = value_frequency[
        value_frequency > threshold * value_frequency.mean()
    ].index

    return data['minutes'].isin(daylight_minutes)
